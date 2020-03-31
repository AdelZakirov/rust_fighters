import torch.nn.parallel
from torch.utils.data import DataLoader
from argparse import Namespace
from pydoc import locate
from dp_tools.experiment_config import experiment_config_cmdline
from dataset import TestSet
from tqdm import tqdm
import pandas as pd

import os
import warnings
warnings.filterwarnings("ignore")


def predict(args: Namespace):
    ind = 0
    for size, checkpoint in zip(args.sizes, args.fine_tune_checkpoints):
        df = pd.DataFrame(columns=['ID', 'leaf_rust', 'stem_rust', 'healthy_wheat'])
        torch.cuda.empty_cache()
        device = torch.device(0)
        cp = torch.load(checkpoint)
        model = locate(cp['model_class'])(**cp['model_hyperparams'])
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(cp['model_state_dict'])

        TEST = os.listdir('Data/test/')
        ds_test = TestSet(TEST, size=size)
        test_loader = DataLoader(dataset=ds_test,
                                 num_workers=args.num_workers,
                                 batch_size=args.batch_size,
                                 shuffle=False)
        predictions = []
        targets = []
        model.eval()
        for batch in tqdm(test_loader):
            if batch is None:
                continue
            with torch.no_grad():
                x, y = batch
                x, y = x.to(device).float(), y.to(device, dtype=torch.int64)
                out = model(x)
                probs = torch.softmax(out.detach().cpu(), dim=1).numpy()
                predictions.extend(list(probs))
                targets.extend(list(y.detach().cpu().numpy()))
        ind = ind + 1
        TEST = [name[:name.find('.')] for name in TEST]
        df['ID'] = TEST
        df[['leaf_rust', 'stem_rust', 'healthy_wheat']] = predictions
        df.to_csv('results/' + args.save_file_name + '_' + str(size) + '_' + str(ind) + '.csv', index=False)


if __name__ == '__main__':
    predict(experiment_config_cmdline())
