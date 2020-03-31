import torch.nn.parallel
from torch.utils.data import DataLoader
from argparse import Namespace
from pydoc import locate
from dp_tools.experiment_config import experiment_config_cmdline
from dataset import DatasetBCE
from loss import  weighted_loss_metric
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def run(args: Namespace):
    df = pd.DataFrame(columns=['ID', 'leaf_rust', 'stem_rust', 'healthy_wheat'])
    names = []
    preds = []
    for fold, checkpoint in zip(args.folds, args.fine_tune_checkpoints):
        torch.cuda.empty_cache()
        device = torch.device(0)
        cp = torch.load(checkpoint)
        model = locate(cp['model_class'])(**cp['model_hyperparams'])
        # model.fc = torch.nn.Linear(in_features=2048, out_features=3)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(cp['model_state_dict'])

        VAL = list(np.loadtxt('Data/kfolds/VAL_CLUST_' + str(fold) + '.txt', delimiter='\n', dtype=str))
        ds_val = DatasetBCE(VAL, mode='val', augmentation=False, size=args.size)
        val_loader = DataLoader(dataset=ds_val,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=False)

        predictions = []
        targets = []
        model.eval()
        for batch in val_loader:
            if batch is None:
                continue
            with torch.no_grad():
                x, y = batch
                x, y = x.to(device).float(), y.to(device, dtype=torch.int64)
                out = model(x)
                probs = torch.softmax(out.detach().cpu(), dim=1).numpy()
                predictions.extend(list(probs))
                targets.extend(list(y.detach().cpu().numpy()))

        names.extend(ds_val.names)
        preds.extend(predictions)
    # preds = np.array(preds)
    df['ID'] = names
    df[['leaf_rust', 'stem_rust', 'healthy_wheat']] = preds
    df.to_csv('results/oof/oof_' + args.save_file_name + '.csv')


if __name__ == '__main__':
    run(experiment_config_cmdline())
