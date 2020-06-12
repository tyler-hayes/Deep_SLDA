import torch
import json
import time
import os
import argparse
from torchvision.models import resnet18

from SLDA_Model import StreamingLDA
import utils
import retrieve_any_layer


def get_feature_extraction_model(ckpt_file, imagenet_pretrained=False):
    feature_extraction_model = resnet18(pretrained=imagenet_pretrained)

    if ckpt_file is not None:
        resumed = torch.load(ckpt_file)
        if 'state_dict' in resumed:
            state_dict_key = 'state_dict'
        else:
            state_dict_key = 'model_state'
        print("Resuming from {}".format(ckpt_file))
        utils.safe_load_dict(feature_extraction_model, resumed[state_dict_key])
    return feature_extraction_model


def pool_feat(features):
    feat_size = features.shape[-1]
    num_channels = features.shape[1]
    features2 = features.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x num_channels
    features3 = torch.reshape(features2, (features.shape[0], feat_size * feat_size, num_channels))
    feat = features3.mean(1)  # mb x num_channels
    return feat


def predict(model, val_data, num_classes=1000):
    num_samples = len(val_data.dataset)
    probabilities = torch.empty((num_samples, num_classes))
    labels = torch.empty(num_samples).long()
    start = 0
    for X, y in val_data:
        # extract feature from pre-trained model and mean pool
        if feature_extraction_wrapper is not None:
            feat = feature_extraction_wrapper(X.cuda())
            feat = pool_feat(feat)
        else:
            feat = X.cuda()
        end = start + feat.shape[0]
        probas = model.predict(feat.cuda(), return_probas=True)
        probabilities[start:end] = probas
        labels[start:end] = y.squeeze()
        start = end
    return probabilities, labels


def get_data_loader(images_dir, training, min_class, max_class, batch_size=128, shuffle=False, dataset='imagenet'):
    if dataset == 'imagenet':
        return utils.get_imagenet_loader(images_dir, min_class, max_class, training, batch_size=batch_size,
                                         shuffle=shuffle)
    else:
        #### IMPLEMENT ANOTHER DATASET HERE ####
        raise NotImplementedError('Please implement another dataset.')


def compute_accuracies(loader, classifier, num_classes):
    probas, y_test_init = predict(classifier, loader, num_classes)
    top1, top5 = utils.accuracy(probas, y_test_init, topk=(1, 5))
    return probas, top1, top5


def update_accuracies(images_dir, curr_max_class, num_classes, classifier, accuracies, save_dir, batch_size, shuffle,
                      dataset):
    seen_classes_test_loader = get_data_loader(images_dir, False, 0, curr_max_class, batch_size=batch_size,
                                               shuffle=shuffle, dataset=dataset)
    seen_probas, seen_top1, seen_top5 = compute_accuracies(seen_classes_test_loader, classifier, num_classes)

    print('\nSeen Classes (%d-%d): top1=%0.2f%% -- top5=%0.2f%%' % (0, curr_max_class - 1, seen_top1, seen_top5))
    accuracies['seen_classes_top1'].append(float(seen_top1))
    accuracies['seen_classes_top5'].append(float(seen_top5))

    # save accuracies and predictions out
    utils.save_accuracies(accuracies, min_class_trained=0, max_class_trained=curr_max_class, save_path=save_dir)
    utils.save_predictions(seen_probas, 0, curr_max_class, save_dir)


def run_experiment(dataset, images_dir, save_dir, classifier, feature_extraction_wrapper, feature_size, batch_size,
                   shuffle, num_classes, class_increment):
    start_time = time.time()
    # start list of accuracies
    accuracies = {'seen_classes_top1': [], 'seen_classes_top5': []}

    first_time = True  # true for base init stage
    slda_save_name = "slda_model_weights_min_trained_0_max_trained_%d"

    # loop over all data and compute accuracy after every "batch"
    for curr_class_ix in range(0, num_classes, class_increment):
        print("\nTraining classes from {} to {}".format(curr_class_ix, curr_class_ix + class_increment))
        max_class = curr_class_ix + class_increment

        # get training loader for current batch
        train_loader = get_data_loader(images_dir, True, curr_class_ix, max_class, batch_size=batch_size,
                                       shuffle=shuffle, dataset=dataset)
        if first_time:
            print('\nGetting data for base initialization...')

            # initialize arrays for base init data because it must be provided all at once to SLDA
            base_init_data = torch.empty((len(train_loader.dataset), feature_size))
            base_init_labels = torch.empty(len(train_loader.dataset)).long()

            # put features into array since base init needs all features at once
            start = 0
            for batch_x, batch_y in train_loader:
                print('\rLoading features %d/%d.' % (start, len(train_loader.dataset)), end='')
                # get feature in real-time
                if feature_extraction_wrapper is not None:
                    # extract feature from pre-trained model and mean pool
                    batch_x_feat = feature_extraction_wrapper(batch_x.cuda())
                    batch_x_feat = pool_feat(batch_x_feat)
                else:
                    batch_x_feat = batch_x.cuda()
                end = start + batch_x_feat.shape[0]
                base_init_data[start:end] = batch_x_feat
                base_init_labels[start:end] = batch_y.squeeze()
                start = end

            # fit base initialization stage
            print('\nFirst time...doing base initialization...')
            classifier.fit_base(base_init_data, base_init_labels)
            first_time = False
        else:
            iters = int(len(train_loader.dataset) / batch_size)
            # fit model
            for batch_ix, (batch_x, batch_y) in enumerate(train_loader):
                # print('\rFitting %d/%d.' % (batch_ix, iters), end='')
                # get feature in real-time
                if feature_extraction_wrapper is not None:
                    # extract feature from pre-trained model and mean pool
                    batch_x_feat = feature_extraction_wrapper(batch_x.cuda())
                    batch_x_feat = pool_feat(batch_x_feat)
                else:
                    batch_x_feat = batch_x.cuda()

                # fit SLDA one example at a time
                for x, y in zip(batch_x_feat, batch_y):
                    classifier.fit(x.cpu(), y.view(1, ))

        # output accuracies to console and save out to json file
        update_accuracies(images_dir, max_class, num_classes, classifier, accuracies, save_dir, batch_size, shuffle,
                          dataset)
        classifier.save_model(save_dir, slda_save_name % max_class)

    # print final accuracies and time
    test_loader = get_data_loader(images_dir, False, 0, num_classes, batch_size=batch_size, shuffle=shuffle,
                                  dataset=dataset)
    probas, y_test = predict(classifier, test_loader, num_classes)
    top1, top5 = utils.accuracy(probas, y_test, topk=(1, 5))
    classifier.save_model(save_dir, "slda_model_weights_final")
    end_time = time.time()
    print('\nFinal: top1=%0.2f%% -- top5=%0.2f%%' % (top1, top5))
    print('\nTotal Time (seconds): %0.2f' % (end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment parameters
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet'])
    parser.add_argument('--images_dir', type=str)  # path to images (folder with 'train' and 'val' for imagenet)
    parser.add_argument('--save_dir', type=str, default=None)  # path to save out results
    parser.add_argument('--expt_name', type=str, default='slda_imagenet')  # name for experiment (& save directory)
    parser.add_argument('--base_init_ckpt',
                        default='./imagenet_files/imagenet_100_class_ckpt.pth')  # need ckpt on base 100 classes for imagenet

    parser.add_argument('--num_classes', type=int, default=1000)  # total number of classes in the dataset
    parser.add_argument('--batch_size', type=int, default=256)  # batch size for getting features & testing
    parser.add_argument('--input_feature_size', type=int, default=512)  # resnet-18 feature size
    parser.add_argument('--shuffle_data', action='store_true')  # true to shuffle data (usually don't want this)
    parser.add_argument('--class_increment', type=int, default=100)  # how many classes before evaluation

    # SLDA parameters
    parser.add_argument('--streaming_update_sigma', action='store_true')  # true to update covariance online
    parser.add_argument('--shrinkage', type=float, default=1e-4)  # shrinkage for SLDA

    args = parser.parse_args()
    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    if args.save_dir is None:
        args.save_dir = 'streaming_experiments/' + args.expt_name

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # setup SLDA model
    classifier = StreamingLDA(args.input_feature_size, args.num_classes, test_batch_size=args.batch_size,
                              shrinkage_param=args.shrinkage, streaming_update_sigma=args.streaming_update_sigma)

    # setup feature extraction model if features are needed on the fly
    if args.base_init_ckpt is not None:
        feature_extraction_model = get_feature_extraction_model(args.base_init_ckpt, imagenet_pretrained=False).eval()
        feature_extraction_wrapper = retrieve_any_layer.ModelWrapper(feature_extraction_model.cuda(), ['layer4.1'],
                                                                     return_single=True).eval()
    else:
        feature_extraction_wrapper = None

    # run the streaming experiment
    run_experiment(args.dataset, args.images_dir, args.save_dir, classifier, feature_extraction_wrapper,
                   args.input_feature_size, args.batch_size, args.shuffle_data, args.num_classes, args.class_increment)
