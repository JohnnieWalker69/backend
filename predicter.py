import os  
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import base64

from datetime import datetime
class BrainTumorClassifier():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.log_path = datetime.now().strftime("%I-%M-%S_%p_on_%B_%d,_%Y")

    def restore_model(self, path):
        
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.model.to(self.device)

class DynamicUNet(nn.Module):
    def __init__(self, filters, input_channels=1, output_channels=1):
        super(DynamicUNet, self).__init__()
        if len(filters) != 5:
            raise Exception(f"Filter list size {len(filters)}, expected 5!")
        padding = 1
        ks = 3
        self.conv1_1 = nn.Conv2d(input_channels, filters[0], kernel_size=ks, padding=padding)
        self.conv1_2 = nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2_1 = nn.Conv2d(filters[0], filters[1], kernel_size=ks, padding=padding)
        self.conv2_2 = nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3_1 = nn.Conv2d(filters[1], filters[2], kernel_size=ks, padding=padding)
        self.conv3_2 = nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv4_1 = nn.Conv2d(filters[2], filters[3], kernel_size=ks, padding=padding)
        self.conv4_2 = nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.maxpool4 = nn.MaxPool2d(2)
        self.conv5_1 = nn.Conv2d(filters[3], filters[4], kernel_size=ks, padding=padding)
        self.conv5_2 = nn.Conv2d(filters[4], filters[4], kernel_size=ks, padding=padding)
        self.conv5_t = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.conv6_1 = nn.Conv2d(filters[4], filters[3], kernel_size=ks, padding=padding)
        self.conv6_2 = nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.conv6_t = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.conv7_1 = nn.Conv2d(filters[3], filters[2], kernel_size=ks, padding=padding)
        self.conv7_2 = nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.conv7_t = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.conv8_1 = nn.Conv2d(filters[2], filters[1], kernel_size=ks, padding=padding)
        self.conv8_2 = nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.conv8_t = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.conv9_1 = nn.Conv2d(filters[1], filters[0], kernel_size=ks, padding=padding)
        self.conv9_2 = nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)
        self.conv10 = nn.Conv2d(filters[0], output_channels, kernel_size=ks, padding=padding)

    def forward(self, x):
        conv1 = F.relu(self.conv1_1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.maxpool1(conv1)
        conv2 = F.relu(self.conv2_1(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.maxpool2(conv2)
        conv3 = F.relu(self.conv3_1(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.maxpool3(conv3)
        conv4 = F.relu(self.conv4_1(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.maxpool4(conv4)
        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))
        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)
        conv6 = F.relu(self.conv6_1(up6))
        conv6 = F.relu(self.conv6_2(conv6))
        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)
        conv7 = F.relu(self.conv7_1(up7))
        conv7 = F.relu(self.conv7_2(conv7))
        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)
        conv8 = F.relu(self.conv8_1(up8))
        conv8 = F.relu(self.conv8_2(conv8))
        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)
        conv9 = F.relu(self.conv9_1(up9))
        conv9 = F.relu(self.conv9_2(conv9))
        output = F.sigmoid(self.conv10(conv9))
        return output

class Api:
    def __init__(self):
        self.device = torch.device('cpu')

    def call(self, file, ofp):
        model = self._load_model()
        save_path = None
        if file != None:
            image = self._get_file(file)
            output = self._get_model_output(image, model)

            name, extension = file.split('.')
            save_path = name+'_predicted'+'.'+extension
            if ofp:
                save_path = os.path.join(ofp,save_path)
            self._save_image(output, save_path)  
            return output 
        
    def _load_model(self):
        filter_list = [16, 32, 64, 128, 256]

        model = DynamicUNet(filter_list).to(self.device)
        classifier = BrainTumorClassifier(model, self.device)
        model_path = './UNet-[16, 32, 64, 128, 256].pt'
        classifier.restore_model(model_path)
        return model

    def _get_model_output(self, image, model):
        """Returns the saved model output"""
        image = image.view((-1, 1, 512, 512)).to(self.device)
        output = model(image).detach().cpu()
        output = (output > 0.5)
        output = output.numpy()
        output = np.resize((output * 255), (512, 512))
        return output

    def _save_image(self, image, path):
        """Save the image to storage specified by path"""
        image = Image.fromarray(np.uint8(image), 'L')
        image.save(path)

    def _get_file(self, file_name):
        """Load the image by taking file name as input"""
        default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512))
        ])

        image = default_transformation(Image.open(file_name))
        return TF.to_tensor(image)



from keras.models import load_model
import cv2
def load_image(filename):
  img = cv2.imread(filename)
  img = cv2.resize(img, (150,150))
  return img
def resolve_label(marker):
  if marker[0] == 0:
    return "Tumor Positive"
  else:
    return "Tumor Negative"
def driver(filename):
  api = Api()
  image = api.call(filename,r"C:\Users\susha\Desktop\backend\images")
  model = load_model('./model.h5')
  name,extension = filename.split('.')
  nf = name + "_predicted" + "." + extension
  data = load_image(nf)
  name, extension = filename.split('.')
  save_path = name+'_predicted'+'.'+extension
  image = Image.fromarray(np.uint8(image), 'L')
  image = base64.b64encode(image.tobytes()).decode('utf-8')
    
#   os.remove(nf)
  return (save_path,resolve_label(np.argmax(model.predict(np.array([data])),axis=1)),image)

# print(driver(r"C:\Users\susha\Desktop\backend\images\download_1.jpeg"))