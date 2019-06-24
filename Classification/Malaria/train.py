import os
import time
import datetime

import numpy as np
import torch
import torch.nn
import torch.nn.functional

import torchvision.datasets.folder

from models.networks import VGG, VGG_AE
from data_utils.data_util import load_image_data, load_folder_data, save_outputs

if __name__ == '__main__':
    ## Parameters
    mode = 'VGG'
    num_models = 1
    t = datetime.datetime.now()
    date = '{}-{:02d}-{:02d}_{:02d}{:02d}'.format(t.year, t.month, t.day, t.hour, t.minute)
    folder = mode + '__' + date
    root = '/media/yaniv/data2/datasets/Malaria'
    test = True
    train_val_rate = 0.8
    batch_size = 64
    epochs = 500

    # Define model
    print('--- Building model -------')
    if mode == 'VGG':
        model = [VGG(32, num_classes=1).cuda() for _ in range(num_models)]
    else:
        num_model = 1
        model = [VGG_AE(32, num_classes=1).cuda() for _ in range(num_models)]
    optimizer = [torch.optim.Adam(model[i].parameters(), lr=0.001) for i in range(len(model))]
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion_ae = torch.nn.L1Loss()
    lambda_AE = 1.0

    # Load data_utils
    print('--- Loading data ---------')
    data_train, data_validation, data_test = load_folder_data(root, train_val_rate, batch_size, test)

    # Train
    print('--- Training -------------')
    best_loss = 0.12
    for t in range(1, epochs*num_models+1):
        m = t % num_models
        running_time = time.perf_counter()
        running_loss = 0
        running_loss_ae = 0
        train_size = 0

        # Train
        model[m].train()
        if mode == 'VGG_EA':
            for batch_idx, batch_data in enumerate(data_validation, 1):
                images, labels = batch_data['images'].cuda(), batch_data['labels'].cuda()

                _, recon = model[m](images, m1=False, m2=True)
                loss_ae_val = criterion_ae(recon, images) * 49152  # Compute the loss

                # Do step at the end to keep that it will be done after validation
                optimizer[m].zero_grad()
                (lambda_AE * loss_ae_val).backward()  # Compute the gradient for each variable
                optimizer[m].step()  # Update the weights according to the computed gradient

        for batch_idx, batch_data in enumerate(data_train, 1):
            images, labels = batch_data['images'].cuda(), batch_data['labels'].cuda()
            if labels.shape[0] < batch_size:
                continue

            if mode == 'VGG':
                outputs = model[m](images)
                loss_ae = 0
            else:
                outputs, recon = model[m](images, m1=True, m2=True)
                loss_ae = criterion_ae(recon, images) * 49152  # images.shape[1] * images.shape[2] * images.shape[3]
                running_loss_ae += loss_ae.item() * labels.shape[0]

            loss = criterion(outputs, labels)  # Compute the loss
            running_loss += loss.item() * labels.shape[0]
            train_size += labels.shape[0]

            if False and mode == 'VGG_AE':
                try:
                    validation_batch = next(validation_sampler)
                    if validation_batch['images'] < batch_size:
                        raise()
                except:
                    validation_sampler = iter(data_validation)
                    validation_batch = next(validation_sampler)
                images, labels = validation_batch['images'].cuda(), validation_batch['labels'].cuda()
                _, recon = model[m](images, m1=False, m2=True)
                loss_ae_val = criterion_ae(recon, images) * 49152
            else:
                loss_ae_val = 0

            # Do step at the end to keep that it will be done after validation
            optimizer[m].zero_grad()
            (loss + lambda_AE * (loss_ae + loss_ae_val)).backward()  # Compute the gradient for each variable
            optimizer[m].step()  # Update the weights according to the computed gradient

        # Validation
        if m == 0: # full round of train on all models
            validation_loss = 0
            validation_loss_ae = 0
            validation_size = 0

            for m_v in range(num_models):
                model[m_v].eval()
            for batch_idx, batch_data in enumerate(data_validation, 1):
                images, labels = batch_data['images'].cuda(), batch_data['labels']

                outputs_v = torch.empty([labels.shape[0], 1, 0], dtype=torch.float)
                for m_v in range(num_models):
                    if mode == 'VGG':
                        with torch.no_grad():
                            outputs = model[m_v](images)
                    else:
                        with torch.no_grad():
                            outputs, recon = model[m_v](images, m1=True, m2=True)
                        validation_loss_ae += criterion_ae(recon, images).item() * 49152 * labels.shape[0]

                    # loss = criterion(outputs, labels)  # Compute the loss
                    # loss_v_i = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels, reduction='none').cpu()
                    # print(loss.shape)
                    # print(loss_v.shape)
                    # loss_v = torch.cat([loss_v, loss_v_i.unsqueeze(2)], dim=2)
                    outputs_v = torch.cat([outputs_v, torch.sigmoid(outputs).cpu().unsqueeze(2)], dim=2)

                outputs_v = torch.where(outputs_v.mean(dim=2) > 0.5, outputs_v.max(dim=2)[0], outputs_v.min(dim=2)[0])
                loss_v = torch.nn.functional.binary_cross_entropy(outputs_v, labels)
                # print(loss_v)
                validation_loss += loss_v * labels.shape[0]
                validation_size += labels.shape[0]
                # validation_loss /= num_models
                # validation_size /= num_models

            running_loss /= train_size
            running_loss_ae /= train_size
            validation_loss /= validation_size
            validation_loss_ae /= validation_size
            print('TIME: {:6.3f} ---- Epoch {}: train loss: {:6f} validation_loss: {:6f}'
                  .format(time.perf_counter() - running_time, t, running_loss, validation_loss))

            test_loss_ae = 0
            test_n = 1
            if test and validation_loss < best_loss:
                print('Saving new best loss')
                best_loss = validation_loss

                test_results = np.ndarray([len(os.listdir(os.path.join(root, 'test'))), 1])
                test_n = 0
                for batch_idx, batch_data in enumerate(data_test, 1):
                    images, labels = batch_data['images'].cuda(), batch_data['labels']

                    outputs_t = torch.empty([labels.shape[0], 1, 0], dtype=torch.float)
                    for m_v in range(num_models):
                        if mode == 'VGG':
                            with torch.no_grad():
                                outputs = model[m_v](images)
                        else:
                            with torch.no_grad():
                                outputs, recon = model[m_v](images, m1=True, m2=True)

                        test_loss_ae += criterion(outputs.cpu(), labels).item() * 49152 * labels.shape[0] # Compute the loss
                        #validation_loss += loss.item() * labels.shape[0]
                        #validation_size += labels.shape[0]

                        outputs_t = torch.cat([outputs_t, torch.sigmoid(outputs).cpu().unsqueeze(2)], dim=2)

                    outputs_t = torch.where(outputs_t.mean(dim=2) > 0.5, outputs_t.max(dim=2)[0], outputs_t.min(dim=2)[0])
                    test_results[test_n:test_n+images.shape[0]] = outputs_t.numpy()
                    test_n += images.shape[0]

                test_n *= num_models

                if not os.path.isdir(os.path.join(root, folder)):
                    os.mkdir(os.path.join(root, folder))
                suffix = os.path.join(folder, 'results-{:.3f}'.format(validation_loss).replace('.', '_'))
                save_outputs(test_results, root, suffix + '.csv')

            if mode == 'VGG_AE':
                test_loss_ae = test_loss_ae / test_n
                print('-------- AE LOSS: TRAIN: {:6.3f} VALIDATION: {:6.3f} TEST: {:6.3f}'
                      .format(running_loss_ae, validation_loss_ae, test_loss_ae))
