import numpy as np
import torch
from tqdm import tqdm as tqdm


def retrieval_image(model, testset, number_of_output=3, batch_size=32):
    test_queries = testset.get_test_queries()

    all_imgs = []
    all_captions = []
    all_queries = []

    # compute test query features
    imgs = []
    mods = []
    for t in tqdm(test_queries):
        imgs += [testset.get_img(t['source_img_id'])]
        mods += [t['mod']['str']]
        if len(imgs) >= batch_size or t is test_queries[-1]:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float()
            imgs = torch.autograd.Variable(imgs).cuda()
            f = model.compose_img_text(imgs, mods).data.cpu().numpy()
            all_queries += [f]
            imgs = []
            mods = []
    all_queries = np.concatenate(all_queries)

    # compute all image features
    imgs = []
    all_processed_images = []
    for i in tqdm(range(len(testset.imgs))):
        imgs += [testset.get_img(i)]
        all_processed_images += [testset.get_img(i)]
        if len(imgs) >= batch_size or i == len(testset.imgs) - 1:
            if 'torch' not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float()
            imgs = torch.autograd.Variable(imgs).cuda()
            imgs = model.extract_img_feature(imgs).data.cpu().numpy()
            all_imgs += [imgs]
            imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]
    all_processed_images = np.array(all_processed_images)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    # match test queries to target images, get nearest neighbors
    nn_result = []
    for i in tqdm(range(all_queries.shape[0])):
        sims = all_queries[i:(i+1), :].dot(all_imgs.T)
        sims[0, test_queries[i]['source_img_id']] = -10e10  # remove query image

        nn_result.append(np.argsort(-sims[0, :])[:110])

    # compute recalls
    nn_result = [[all_captions[nn] for nn in nns][:number_of_output] for nns in nn_result]
    nn_image_result = [[all_processed_images[nn] for nn in nns][:number_of_output] for nns in nn_result]

    return nn_result, nn_image_result