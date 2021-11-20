import random
import torch.nn.functional as F
from utils.create_file import *
from utils.picture_process import *
import cv2
from time import sleep


class Model(torch.nn.Module):
    def __init__(self, cates):
        super(Model, self).__init__()
        self.cates = cates
        # self.conv1 = torch.nn.Conv2d(3, 15, kernel_size=(25, 25))
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=(5, 5), padding='same')  # 3 100 100

        # self.mp1 = torch.nn.MaxPool2d(kernel_size=(4, 4))
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 3 50 50

        # self.conv2 = torch.nn.Conv2d(15, 30, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=(3, 3), padding=1)  # 6 50 50

        # self.mp2 = torch.nn.MaxPool2d(kernel_size=(3, 3))
        self.mp2 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 6 25 25

        self.conv3 = torch.nn.Conv2d(6, 8, kernel_size=(2, 2))  # 8 24 24

        self.mp3 = torch.nn.MaxPool2d(kernel_size=(2, 2))  # 8 12 12

        # self.linear = torch.nn.Linear(750, 4)
        self.linear1 = torch.nn.Linear(1152, 384)
        self.linear2 = torch.nn.Linear(384, 192)
        # self.linear3 = torch.nn.Linear(5000, 1000)
        self.linear4 = torch.nn.Linear(192, 96)
        self.linear5 = torch.nn.Linear(96, self.cates)

    def forward(self, x):
        print(x.size)
        in_size = x.size(0)
        # print(x.shape)
        # print(x.size)
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = F.relu(self.conv3(x))
        x = self.mp3(x)
        # print(x.shape)
        x = x.view(in_size, -1)
        # print(x.shape)
        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)

        return torch.sigmoid(x)


def frame_to_tensor(frame):
    img_np = np.array(frame).reshape((-1, 100, 100))
    # print(img_np.shape)
    print(img_np.shape)
    img_np = img_np.transpose((2, 0, 1))
    img_tensor = torch.from_numpy(img_np).view(1, -1, 100, 100) / 255.0
    return img_tensor


def train(num_epochs, train_loader):
    total_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        input, label = data
        output = net(input)
        loss = criterion(output, label)

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


if __name__ == '__main__':
    # path
    train_data_file = './train_data_file/'
    Users_img_file = './Users_img_file/'
    cates = 0
    for i in os.listdir(Users_img_file):
        cates += 1

    criterion = torch.nn.CrossEntropyLoss()

    # choice = 0
    choice = int(input('0 or 1'))
    if choice == 0:
        net = Model(cates + 1)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        user_name = input('请输入姓名：')
        # 创建user_file
        user_file_name = create_user_file(Users_img_file, user_name)  # 0_ZT
        counter = 0
        capture = cv2.VideoCapture(0)
        face_model = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        while 1:
            ret, frame = capture.read()

            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            face_location = face_model.detectMultiScale(grey)
            param = '--ing'
            for (x, y, w, h) in face_location:
                cv2.rectangle(img=frame, pt1=(x - 10, y - 10), pt2=(x + w + 10, y + h + 10), color=(0, 255, 0),
                              thickness=3)

                grey1 = Image.fromarray(grey)
                grey_img = grey1.crop((x - 10, y - 10, x + w + 10, y + h + 10))

                if counter < 1:
                    # (left, upper, right, lower)-tuple

                    idx = random.randint(1000, 10000)
                    grey_img.save(Users_img_file + user_file_name + '/' + user_file_name + str(idx) + '.jpg')
                    counter += 1

                    print(counter)
                    sleep(2)

                else:
                    print('Now training')
                    image_process = ImageProcess(Users_img_file, train_data_file)
                    image_process.resize_img()
                    train_loader = image_process.loader()
                    # print('1')
                    for i in os.listdir(train_data_file):
                        os.remove(train_data_file + i)
                    # print('2')
                    train(25, train_loader=train_loader)
                    print('train over')

                    predicted_grey = grey_img.resize((100, 100))
                    predicted_grey = predicted_grey.convert('L')
                    predicted = net(frame_to_tensor(predicted_grey)).sum().item()
                    param_num = int(float(str(predicted)))
                    print(param_num, param_num)
                    for user_name in os.listdir(Users_img_file):
                        if int(user_name.split('_')[0]) == param_num:
                            param = user_name.split('_')[1]

                cv2.putText(frame, param, (x - 20, y - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.2,
                            color=(255, 0, 0), thickness=2, )
                cv2.imshow('face_detection', frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    else:
        print('Now training')

        net = Model(cates)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        # user_name = input('请输入姓名：')
        # 创建user_file
        # user_file_name = create_user_file(Users_img_file, user_name)  # 0_ZT
        # counter = 0

        # print('1')
        for i in os.listdir(train_data_file):
            os.remove(train_data_file + i)
        # print('2')
        capture = cv2.VideoCapture(0)
        face_model = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

        image_process = ImageProcess(Users_img_file, train_data_file)
        image_process.resize_img()
        train_loader = image_process.loader()
        train(25, train_loader=train_loader)
        print('train over')
        param = 'NONE'

        while 1:
            ret, frame = capture.read()

            grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            face_location = face_model.detectMultiScale(grey)
            # param = '--ing'
            for (x, y, w, h) in face_location:
                cv2.rectangle(img=frame, pt1=(x - 10, y - 10), pt2=(x + w + 10, y + h + 10), color=(0, 255, 0),
                              thickness=3)

                grey1 = Image.fromarray(grey)
                # grey_img = grey1.crop((x - 10, y - 10, x + w + 10, y + h + 10))

                predicted_grey = grey1.resize((100, 100))
                predicted_grey = predicted_grey.convert('L')
                predicted = net(frame_to_tensor(predicted_grey)).sum().item()
                param_num = int(float(str(predicted)))
                print(param_num, param_num)
                for user_name in os.listdir(Users_img_file):
                    if int(user_name.split('_')[0]) == param_num:
                        param = user_name.split('_')[1]
                cv2.putText(frame, param, (x - 20, y - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.2,
                            color=(255, 0, 0), thickness=2, )
                cv2.imshow('face_detection', frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
