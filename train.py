import models
import util


def train_model(model, criterion, optimizer, lr_scheduler, path, num_epochs=200, model_name='lstm'):
    since = time.time()
    
    best_model = model
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in [train_np_dir, test_np_dir]:
            if phase == train_np_dir:
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            counter = 0
            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                
                inputs, labels = data
                
                # wrap them in Variable
                if use_gpu:
                    #                     print(inputs)
                    inputs = Variable(inputs.view(-1, sequence_length, input_size).cuda())
                    labels = Variable(labels.view(-1).cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
            
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                #                 print (inputs.size())
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                #                 print (labels, outputs)
                
                # backward + optimize only if in training phase
                if phase == train_np_dir:
                    loss.backward()
                    optimizer.step()
                
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                    
                    #             print(preds, labels.data, '\n========')
                epoch_loss = running_loss / dset_sizes[phase]
                epoch_acc = running_corrects / dset_sizes[phase]
                            
                print('running corrects: ', running_corrects)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                                    
                # deep copy the model
                if phase == test_np_dir and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model = copy.deepcopy(model)
                    # saving weights
                    torch.save(model, data_dir + '/' + trained_weights_dir + "/" + model_name + '_' + str(epoch) + ".pt")

                                                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

os.listdir()

data_dir = 'data'
train_np_dir = 'train_np'
test_np_dir = 'test_np'
trained_weights_dir = 'trained_weights'

# Load existing clasess
pkl_file = open('classes_dict.pickle', 'rb')

classes_dict= pickle.load(pkl_file)

pkl_file.close()
num_classes = len(classes_dict)
input_size = 512
print(classes_dict['PlayingGuitar'], len(classes_dict))

opt = {
    'hidden_size' : 256,
    'input_size' : 512,
    'num_layers' : 2,
    'dropout' : 0.8,
    'batch_size' : 2,
    'num_epochs' : 200,
    'learning_rate' : 0.1,
    'sequence_length' : 50
}

use_gpu = torch.cuda.is_available()
print ("GPU is available: ", use_gpu)

model = ConvLSTM(input_size = input_size, hidden_size = opt['hidden_size'],
                      num_layers = opt['num_layers'], num_classes = num_classes, dropout = opt['dropout']).cuda()
rnn_batch

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=opt['learning_rate'], weight_decay=0)

dset_loaders = {x: get_loader(data_dir + '/' + x)[0] for x in [train_np_dir, test_np_dir]}

dset_sizes = {train_np_dir: train_size, test_np_dir: test_size}
print(dset_sizes)


model_lstm = train_model(model, criterion, optimizer, exp_lr_scheduler, '', num_epochs=opt['num_epochs'], model_name='lstm')

for data in get_loader(data_dir + '/' + test_np_dir, batch_size=3000)[0]:
    # get the inputs
    
    inputs, labels = data
    
    # wrap them in Variable
    if use_gpu:
        x_test = Variable(inputs.view(-1, sequence_length, input_size).cuda())
        y_test = labels.view(-1).cpu().numpy()
        y_pred = model_lstm(x_test)
        _, y_pred = torch.max(y_pred.data, 1)
        y_pred = y_pred.cpu().view(-1).numpy()
    break

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()
class_int_to_name = {classes_dict[key]:key for key in classes_dict}
names_of_classes = [class_int_to_name[i] for i in range(0, num_classes)]
print(len(names_of_classes), len(y_test))
# plot_confusion_matrix(cnf_matrix, classes=names_of_classes,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
# plt.figure()
plot_confusion_matrix(cnf_matrix, classes=names_of_classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
