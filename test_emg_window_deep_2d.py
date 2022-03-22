from sklearn.metrics import confusion_matrix, classification_report
from emg_pytorch_dataset import EmgDatasetMap
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from emg_pytorch_model import RawEmgConvnet
import utils
import os
import logging


def test_window(model, dataloader, device, logger, base_class_num=0):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct_test = 0
    total = 0
    running_loss = 0
    counter = 0
    y_pred = []
    y_labels = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            batch_size = data[1].size(0)
            if batch_size == 1:
                continue
            counter += 1
            emg, labels = data
            emg = emg.to(device)
            labels = (labels - base_class_num).to(device)
            outputs = model(emg)
            loss = criterion(outputs, labels)
            running_loss += float(loss)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            y_pred += (predicted + base_class_num).cpu().tolist()
            y_labels += (labels + base_class_num).cpu().tolist()
    los = running_loss / float(counter)
    acc = 100 * correct_test / total
    logger.info(f'Accuracy of the network on the {total} test windows: %d %%' % acc)
    logger.info(f'Loss of the network on the {total} test windows: %d %%' % los)
    logger.info('\n' + classification_report(y_true=y_labels, y_pred=y_pred))
    logger.info(f'\n{confusion_matrix(y_true=y_labels, y_pred=y_pred)}')
    return los, acc

def main():
    logger = utils.config_logger(os.path.basename(__file__)[:-3], level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filter = lambda df: df.iloc[500000:650000, :]
    dataset = EmgDatasetMap(users_list=['06'], data_dir=utils.HDF_FILES_DIR, window_size=30, filter_fn=filter)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    model = RawEmgConvnet(number_of_class=11, enhanced=False)
    model.load_state_dict(torch.load('./window_2d.pt'))
    model.to(device)
    test_window(model, dataloader, device, logger)


if __name__ == '__main__':
    main()
