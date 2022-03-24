import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from emg_pytorch_dataset import EmgDatasetMap
from test_emg_window_deep_2d import test_window
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from emg_pytorch_model import RawEmgConvnet
from models.model3d import RawEmg3DConvnet
import utils
import os
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


logger = utils.config_logger(os.path.basename(__file__)[:-3], level=logging.DEBUG)
writer = SummaryWriter()
# constants
NUM_OF_EPOCHS = 5
NUM_OF_USERS = 44
BATCH_SIZE = 32
WINDOW_SIZE = 1280
WINDOW_STRIDE = int(WINDOW_SIZE / 64)
NUM_OF_CLASSES = 4
BASE_CLASS_NUM = 6
DEBUG_PRINT_ITERATION = 100
SHRINK_TO_ONE_ROW = False
LEARNING_RATE = 0.000001  # 0.000011288378916846883
MAX_CACHE_SIZE = 20
MODEL_TYPE = RawEmg3DConvnet
assert MODEL_TYPE in [RawEmg3DConvnet, RawEmgConvnet]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
logger.debug(f'Device: {device}')
users_train_list = utils.FULL_USER_LIST[:4]
users_test_list = ['06']
users_train_list = [f for f in users_train_list if f not in users_test_list]
# assert int(len(users_train_list)) + int(len(users_test_list)) == num_of_users, 'Wrong Users Number'
logger.debug(f'User Train List:\n{users_train_list}')
logger.debug(f'User Test List:\n{users_test_list}')


def filter_func(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df['TRAJ_GT'] > BASE_CLASS_NUM - 1, :]


train_dataset = EmgDatasetMap(users_list=users_train_list, data_dir=utils.HDF_FILES_DIR, window_size=WINDOW_SIZE,
                              stride=WINDOW_STRIDE, max_cache_size=MAX_CACHE_SIZE, load_to_memory=True,
                              filter_fn=filter_func, logger=logger)
test_dataset = EmgDatasetMap(users_list=users_test_list, data_dir=utils.HDF_FILES_DIR, window_size=WINDOW_SIZE,
                             stride=WINDOW_STRIDE, load_to_memory=True, max_cache_size=4,
                             filter_fn=filter_func, logger=logger)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model = RawEmg3DConvnet(number_of_classes=NUM_OF_CLASSES, window_size=WINDOW_SIZE).to(device) \
    if MODEL_TYPE == RawEmg3DConvnet else \
    RawEmgConvnet(number_of_class=NUM_OF_CLASSES, enhanced=True, window_size=WINDOW_SIZE,
                  shrink_to_one_raw=SHRINK_TO_ONE_ROW).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train
train_loss_list, train_accuracy_list = [], []
val_loss_list, val_accuracy_list = [], []


epoch_pbar = tqdm(range(NUM_OF_EPOCHS))
total_counter = 0
for epoch in epoch_pbar:
    model.train()
    running_loss, correct_train, total_train, counter = 0, 0, 0, 0
    y_pred, y_labels = [], []
    for i, data in enumerate(train_dataloader, 0):
        batch_size = data[1].size(0)
        if batch_size < BATCH_SIZE:
            continue
        counter += 1
        total_counter += 1
        emg_data = data[0].to(device)
        labels = (data[1] - BASE_CLASS_NUM).to(device)
        total_train += labels.size(0)
        optimizer.zero_grad()
        outputs = model(emg_data)
        loss = criterion(outputs, labels)
        writer.add_scalar("Loss/train", float(loss), total_counter)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        correct_train += int(correct)
        running_loss += float(loss)
        labels += BASE_CLASS_NUM
        acc = float(correct) / float(labels.size(0))
        writer.add_scalar("Accuracy/train", acc, total_counter)
        y_pred += (predicted + BASE_CLASS_NUM).cpu().tolist()
        y_labels += labels.cpu().tolist()
        if i % DEBUG_PRINT_ITERATION == DEBUG_PRINT_ITERATION - 1:
            _, global_counts = torch.tensor(y_labels).unique(return_counts=True)
            unique, counts = labels.cpu().unique(return_counts=True)
            logger.debug(f'Epoch {epoch} batch num {i} loss {float(loss)}'
                         f' accuracy {acc} '
                         f'labels {unique.tolist()}, counts {counts.tolist()}')
            writer.add_scalar('Counts Std/Mean', global_counts.float().std() / global_counts.float().mean()
                              , global_step=total_counter)
            writer.add_scalar('Counts Max/Min', global_counts.float().max() / global_counts.float().min()
                              , global_step=total_counter)
            writer.add_scalar('Curren Counts Std/Mean', counts.float().std() / counts.float().mean()
                              , global_step=total_counter)
            writer.add_scalar('Current Counts Max/Min', counts.float().max() / counts.float().min()
                              , global_step=total_counter)

            # writer.add_graph(model, emg_data)
            # writer.add_embedding(emg_data.reshape(BATCH_SIZE, -1),
            #                      metadata=labels.cpu().tolist(), global_step=total_counter) #label_img=emg_data)

        del loss, outputs, emg_data, labels, predicted, correct
    epoch_loss = running_loss / float(counter)
    train_acc = 100 * correct_train / total_train
    train_loss_list.append(epoch_loss)
    train_accuracy_list.append(train_acc)
    logger.info(f'\nEpoch {epoch}: \n' + classification_report(y_true=y_labels, y_pred=y_pred))
    logger.info(f'\n{confusion_matrix(y_true=y_labels, y_pred=y_pred)}\n')

    # validation
    logger.info(f'Epoch {epoch} Validation')

    val_loss, val_acc = test_window(model, test_dataloader, device, logger,
                                    base_class_num=BASE_CLASS_NUM)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_acc)
    epoch_pbar.set_postfix({'epoch': epoch, 'train loss': epoch_loss, 'train_accuracy': train_acc,
                            'val loss': val_loss, 'val_accuracy': val_acc})
    # epoch_pbar.set_postfix({'epoch': epoch, 'train loss': epoch_loss, 'train_accuracy': train_acc})

    writer.flush()
utils.show_learning_curve(train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, NUM_OF_EPOCHS,
                          title='window 2d loss and accuracy')

# test
# test_dataset = EmgDatasetMap(users_list=users_test_list, data_dir=utils.HDF_FILES_DIR, window_size=WINDOW_SIZE,
#                              stride=WINDOW_STRIDE,
#                              filter=filter_func)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# test_window(model, test_dataloader, device, logger, base_class_num=BASE_CLASS_NUM)
torch.save(model.state_dict(), './window_2d.pt')
