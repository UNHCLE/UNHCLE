### Steps to Run 👋

First download YouCook2 data from http://youcook2.eecs.umich.edu/ and place them in folder named `feat_csv`.

Then run `make_feat.py` for both training and testing/validation files to create the feature files as `.npy` dump.

Run `python3 siam-correct-cook-dtw-comment.py` to start training the UNHCLE model. Models are saved as `.pth` extensions which can be easily saved and loaded using `torch.load()`.

The model is trained for 100 epochs.

For evaluation, set batch_size = 1 and use appropriate feature files in dataloader and training code.
