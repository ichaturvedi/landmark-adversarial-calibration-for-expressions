import torch
import model
import data_jx
import torchvision

def load_saved_model():
    """
    Only for use in remote server, where model is saved under './saved_models'
    """
    model = torch.load('saved_models/generator.pt')

    return model


def eval_model(generator):

    dataFeeder = data_jx.testDataLoader('/export/home/itic/pdt_feb11/jx_data/test_data')
    train_loader = torch.utils.data.DataLoader(dataFeeder, batch_size=1, shuffle=True,
											   num_workers=1, pin_memory=True)

    generator.eval()

    # get some random training images
    dataiter = enumerate(train_loader)
    # img = dataiter.next()
    for i, img in dataiter:
        I3_var = img.to(torch.float32).cuda() 

        fake = generator(I3_var)

        print("saving file")
        output_pair = torch.cat((I3_var, fake),3)
        torchvision.utils.save_image((output_pair+1)/2, '/export/home/itic/pdt_feb11/jx_test_samples/test'+str(i)+'.jpg')
		

if __name__ == '__main__':
    eval_model(load_saved_model())
