# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

import argparse

import numpy as np
import pysam
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(8800, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(self.conv4(x), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 
                len(train_loader) * len(data),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    num_examples_seen = 0
    num_batches_seen = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_test_loss = F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            num_examples_seen += len(data) # len(data) is batch size
            num_batches_seen += 1
            
            test_loss += batch_test_loss / len(data)
    print('num example seen in test: {}'.format(num_examples_seen))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss / num_batches_seen, correct, num_examples_seen,
        100. * correct / num_examples_seen))


def get_images(encoded_data):
    num_classes = encoded_data['single_reads'].max() + 1
    # Load the images
    encoded_batch = encoded_data['single_reads']
    encoded_batch = encoded_batch[:,:30,:] # clip

    #One hot encode them in NCHW format
    encoded_batch = np.eye(num_classes)[encoded_batch].transpose(0,3,1,2).astype(np.float32)
    return encoded_batch


def main():
    parser = argparse.ArgumentParser(description="Train a model for single-read infernece using a A CNN")

    parser.add_argument('--fp',
                        type=str,
                        default='FP.npy',
                        help='non-true variant examples')
    parser.add_argument('--tp',
                        type=str,
                        default='TP.npy',
                        help='True variant examples')
    parser.add_argument('--batch_size',
                        type=int,
                        default=20,
                        help='Batch size for training model')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to run in training')
    parser.add_argument('--num_test',
                        type=int,
                        default=1000,
                        help='Number of samples to use for testing')
    parser.add_argument('--outfile',
                        type=str,
                        default='out',
                        help='VCF or model checkpoint, depending on whether running in training or inference mode')
    parser.add_argument('--infile',
                        type=str,
                        default=None,
                        help='Model checkpoint or model for inference')
    parser.add_argument('--inference',
                        default=False,
                        help='Run in inference mode',
                        action='store_true')
    parser.add_argument('--inference_data',
                        type=str,
                        default=True,
                        help='Encoded reads to run inference on')
    parser.add_argument('--sample_vcf',
                        type=str,
                        default=None,
                        help='VCF from which to get header')

    args = parser.parse_args()


    fp_images = np.load(args.fp)
    tp_images = np.load(args.tp)

    orig_num_fp_samples = len(fp_images) # This many variants
    orig_num_tp_sampels = len(tp_images) # This many variants


    new_num_fp_samples = (orig_num_fp_samples // args.batch_size) * args.batch_size
    new_num_tp_samples = (orig_num_tp_sampels // args.batch_size) * args.batch_size

    print('{} FP samples and {} TP samples'.format(new_num_fp_samples, new_num_tp_samples))
    
    # Trim the number of images so as to fit into batch size multiple
    fp_images = fp_images[:new_num_fp_samples]
    tp_images = tp_images[:new_num_tp_samples]
    
    all_data = np.concatenate((fp_images, tp_images))

    #Extract the raw data from the numpy multiarrays
    fp_batch = get_images(fp_images)
    tp_batch = get_images(tp_images)
    fp_labels = np.zeros(len(fp_images)).astype(int)
    tp_labels = np.ones(len(tp_images)).astype(int)

    # Shuffle the data
    all_batches = np.vstack((fp_batch, tp_batch))
    labels = np.concatenate((fp_labels, tp_labels))
    indices = np.random.permutation(len(labels))
    all_batches_shuffled = all_batches[indices]
    labels_shuffled = labels[indices]
    num_batches = len(all_batches_shuffled) // args.batch_size
    all_batches_shuffled = np.split(all_batches_shuffled, num_batches)
    labels_shuffled = np.split(labels_shuffled, num_batches)

    # Marshall the data into pytorch tensors:
    batches_tensors = [torch.from_numpy(a) for a in all_batches_shuffled]
    labels_tensors = [torch.from_numpy(a) for a in labels_shuffled]


    
    # Get a device, preferably GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    if not args.inference:
        # Run training:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        full_data = list(zip(batches_tensors, labels_tensors))
        training_data = full_data[:-args.num_test // args.batch_size]
        test_data = full_data[-args.num_test // args.batch_size:]    
    
        for epoch in range(1, args.num_epochs + 1):
            train(200, model, device, training_data, optimizer, epoch)
    
        # Test    
        test(model, device, test_data)
    
        torch.save(model.state_dict(), args.outfile)
    else:
        model.load_state_dict(torch.load(args.infile))
        #Run in inferecne mode:
        inference_batch_size = 1000
        num_inference_batches = len(all_batches) // inference_batch_size
        outputs = []
        with torch.no_grad():
            for b in range(num_inference_batches):
                subset = all_batches[b*inference_batch_size:(b + 1) * inference_batch_size]
                data = torch.from_numpy(subset).to(device)
                output = model(data)
                outputs.append(output)
        pred = []
        for output in outputs:
            pred.extend(output.max(1, keepdim=True)[1].reshape(-1).cpu().numpy())
        bcf_in = pysam.VariantFile(args.sample_vcf)  # auto-detect input format
        bcf_out = pysam.VariantFile(args.outfile, 'w', header=bcf_in.header)
        bcf_out.close()
        with open(args.outfile, "a") as f:
            for i, p in enumerate(pred): 
                if p: #variant predicted
                    image = all_data[i]
                    f.write(image['name'])

if __name__ == '__main__':
    main()
