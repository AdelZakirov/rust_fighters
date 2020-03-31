import torch.nn.parallel
from torch.utils.data import DataLoader
from argparse import Namespace
from pydoc import locate
from tqdm import tqdm
import random

from utils.meters import AverageMeter
from utils.logger import get_logger
from utils.helpers import *
from utils.experiment_config import experiment_config_cmdline
from utils.checkpointer import ModelCheckpoint
from dataset import WheatRust
from utils.early_stopping import EarlyStopping
from loss import wcross_entropy as cross_entropy
from loss import weighted_loss_metric
from sklearn.metrics import f1_score


# load the model
def load_model(args, fold=0):
    checkpoint = args.fine_tune_checkpoints[fold]
    device = torch.device('cuda:0')
    if checkpoint:
        cp = torch.load(checkpoint)
        model = locate(cp['model_class'])(**cp['model_hyperparams'])
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(cp['model_state_dict'])
        if 'binary' in args.mode:
            model.module._fc = torch.nn.Linear(in_features=1792, out_features=2, bias=True)
        model = model.to(device)
        print('Continue training from ' + checkpoint)
    else:
        model = locate(args.model_class)(**args.model_hyperparams)
        model = model.to(device)
        model = torch.nn.DataParallel(model).cuda()
        print('Weights initialized from Imagenet')
    optimizer = define_optimizer(model, args)

    if checkpoint:
        optimizer.load_state_dict(cp['optimizer_state_dict'])
    return model, optimizer


# load the dataloaders
def get_dataloaders(args, fold):
    TRAIN = list(np.loadtxt('Data/kfolds/TRAIN_CLUST_' + str(fold) + '.txt', delimiter='\n', dtype=str))
    VAL = list(np.loadtxt('Data/kfolds/VAL_CLUST_' + str(fold) + '.txt', delimiter='\n', dtype=str))
    labels = list(np.loadtxt('Data/kfolds/labels_train_clust' + str(fold) + '.txt', dtype=int))
    if args.mode == 'default':
        ds_train = WheatRust(TRAIN, size=args.size, augmentation=args.augmentation, mode='default',
                             lweight=args.lweight)
        ds_val = WheatRust(VAL, size=args.size, augmentation='none', mode='default', lweight=False)
        weights = get_class_weights(labels)
        sampler_, shuffle_ = define_sampler(args, weights)
    elif args.mode == 'mixed':
        TEST = list(np.loadtxt('Data/TEST_confidant.txt', delimiter='\n', dtype=str))
        ds_pseudo = WheatRust(TEST, size=args.size, augmentation=args.augmentation, mode='pseudo', pweight=args.pweight)
        ds_train = WheatRust(TRAIN, size=args.size, augmentation=args.augmentation, mode='default',
                             lweight=args.lweight)
        ds_val = WheatRust(VAL, size=args.size, augmentation='none', mode='default', lweight=False)
        ds_train = ds_train + ds_pseudo
        sampler_, shuffle_ = None, True
    elif args.mode == 'pseudo':
        TEST = list(np.loadtxt('Data/TEST_confidant.txt', delimiter='\n', dtype=str))
        ds_train = WheatRust(TEST, size=args.size, augmentation=args.augmentation, mode='pseudo', pweight=1.0)
        ds_val = WheatRust(TRAIN + VAL, size=args.size, augmentation='none', mode='default', lweight=False)
        sampler_, shuffle_ = None, True
    elif args.mode == 'binary':
        ds_train = WheatRust(TRAIN, size=args.size, augmentation=args.augmentation, mode='binary', lweight=args.lweight)
        ds_val = WheatRust(VAL, size=args.size, augmentation='none', mode='binary', lweight=False)
        labels = list(np.loadtxt('Data/kfolds/labels_train_clust_hva' + str(fold) + '.txt', dtype=int))
        weights = get_class_weights(labels)
        sampler_, shuffle_ = define_sampler(args, weights)
    elif args.mode == 'pval':
        ds_train = WheatRust(TRAIN + VAL, size=args.size, augmentation=args.augmentation, mode='default',
                             lweight=args.lweight)
        TEST = list(np.loadtxt('Data/TEST_confidant.txt', delimiter='\n', dtype=str))
        ds_val = WheatRust(TEST, size=args.size, augmentation='none', mode='pseudo', pweight=1.0)
        labels = list(np.loadtxt('Data/kfolds/labels_all.txt', dtype=int))
        weights = get_class_weights(labels)
        sampler_, shuffle_ = define_sampler(args, weights)
    elif args.mode == 'binary_rust':
        TRAIN = list(np.loadtxt('Data/kfolds/TRAIN_CLUST_RUST_' + str(fold) + '.txt', delimiter='\n', dtype=str))
        VAL = list(np.loadtxt('Data/kfolds/VAL_CLUST_RUST_' + str(fold) + '.txt', delimiter='\n', dtype=str))
        labels = list(np.loadtxt('Data/kfolds/labels_train_clust_RUST_' + str(fold) + '.txt', dtype=int))
        ds_train = WheatRust(TRAIN, size=args.size, augmentation=args.augmentation, mode='binary_rust',
                             lweight=args.lweight)
        ds_val = WheatRust(VAL, size=args.size, augmentation='none', mode='binary_rust', lweight=False)
        weights = get_class_weights(labels)
        sampler_, shuffle_ = define_sampler(args, weights)
    train_loader = DataLoader(dataset=ds_train,
                              sampler=sampler_,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=shuffle_)
    val_loader = DataLoader(dataset=ds_val,
                            num_workers=args.num_workers,
                            batch_size=8,
                            shuffle=False)
    return train_loader, val_loader


# train function
def train(args, model, device, train_loader, optimizer, scheduler, loss_function, epoch, train_loss, log):
    predictions = []
    targets = []
    model.train()
    for batch in tqdm(train_loader, dynamic_ncols=True, desc=f"epoch {epoch} train"):
        if batch is None:
            continue
        x, y, w = batch
        x, y, w = x.to(device).float(), y.to(device).float(), w.to(device).float()
        if args.mixup:
            x, y_a, y_b, lam = mixup_data(x, y, args.alpha, True)
        out = model(x)
        if args.mixup:
            loss = mixup_criterion(loss_function, w, out, y_a, y_b, lam)
        else:
            loss = loss_function(out, y, w)
        train_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        probs = torch.softmax(out.detach().cpu(), dim=1).numpy()
        predictions.extend(list(probs))
        targets.extend(list(y.detach().cpu().numpy()))

    predictions = np.array(predictions)
    targets = np.array(targets)
    predictions_argmax = predictions.argmax(axis=1)
    targets_argmax = targets.argmax(axis=1)
    F_score_macro = f1_score(targets_argmax, predictions_argmax, average='macro')
    train_log_loss = weighted_loss_metric(targets, predictions)
    print('------TRAIN-------')
    log.info(f"epoch {epoch} train_loss = {train_loss.value:0.3f}")
    log.info(f"epoch {epoch} train_log_loss = {train_log_loss:0.3f}")
    log.info(f"epoch {epoch} train_F_score_macro = {F_score_macro:0.3f}")


def validate(model, device, val_loader, loss_function, epoch, dev_loss, log):
    predictions = []
    targets = []
    model.eval()
    for batch in tqdm(val_loader, dynamic_ncols=True, desc=f"epoch {epoch} dev"):
        if batch is None:
            continue
        with torch.no_grad():
            x, y, w = batch
            w = torch.ones(y.size(0))
            x, y = x.to(device).float(), y.to(device).float()
            w = w.to(device).float()
            out = model(x)
            loss = loss_function(out, y, w)
            dev_loss.append(loss)
            probs = torch.softmax(out.detach().cpu(), dim=1).numpy()
            predictions.extend(list(probs))
            targets.extend(list(y.detach().cpu().numpy()))

    predictions = np.array(predictions)
    targets = np.array(targets)
    predictions_argmax = predictions.argmax(axis=1)
    targets_argmax = targets.argmax(axis=1)
    F_score_macro = f1_score(targets_argmax, predictions_argmax, average='macro')
    dev_log_loss = weighted_loss_metric(targets, predictions)
    print('------VALIDATION-------')
    log.info(f"epoch {epoch} val_loss = {dev_loss.value:0.3f}")
    log.info(f"epoch {epoch} val_log_loss = {dev_log_loss:0.3f}")
    log.info(f"epoch {epoch} val_F_score_macro = {F_score_macro:0.3f}")

    return dev_log_loss


def train_model(args: Namespace):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    for fold in args.folds:
        es = EarlyStopping(patience=args.es_patience)
        print(f'STARTING FOLD {fold}')
        torch.cuda.empty_cache()
        run_name = args.run_name + '_' + str(fold)

        device = torch.device('cuda:0')
        model, optimizer = load_model(args, fold)
        train_loader, val_loader = get_dataloaders(args, fold)
        loss_function = cross_entropy
        scheduler = define_lr_scheduler(args, optimizer, train_loader)

        train_loss = AverageMeter()
        dev_loss = AverageMeter()

        log = get_logger('zindi' + ":" + run_name)
        checkpoint_path = 'checkpoints/' + run_name
        checkpointer = ModelCheckpoint(checkpoint_path, 'checkpoint',
                                       n_saved=5,
                                       score_name='NLL_loss',
                                       save_as_state_dict=False,
                                       require_empty=False)

        for epoch in range(args.max_epochs):
            train_loss.reset()
            dev_loss.reset()

            train(args, model, device, train_loader, optimizer, scheduler, loss_function, epoch, train_loss, log)
            dev_log_loss = validate(model, device, val_loader, loss_function, epoch, dev_loss, log)

            checkpointer({'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'model_class': args.model_class,
                          'model_hyperparams': args.model_hyperparams,
                          'optimizer_state_dict': optimizer.state_dict(),
                          'loss': dev_loss.value},
                         score=-dev_log_loss)

            if es.step(dev_log_loss):
                print('EARLY STOPPING ON EPOCH ' + str(epoch))
                break


if __name__ == '__main__':
    train_model(experiment_config_cmdline())
