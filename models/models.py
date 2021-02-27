import torch
import torch.nn as nn

time_length0 = 10039 ## 10039 is half of the sector time
mid_length0 = (time_length0 - 1)//2

time_length1 = 3345 ## read from the conv output size
mid_length1 = (time_length1 - 1)//2

time_length2 = 1113 ## read from the conv output size
mid_length2 = (time_length2 - 1)//2


class ModelPlain(nn.Module):
    def __init__(self, n=16):
        super(ModelPlain, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv1d(12, n, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv1d(n, n, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv1d(n, n, 9, stride=1, padding=4),
            nn.ReLU())

        self.mp = nn.MaxPool1d(7, stride=3, return_indices=False)

        self.conv1 = nn.Sequential(
            nn.Conv1d(n, 2*n, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv1d(2*n, 2*n, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv1d(2*n, 2*n, 9, stride=1, padding=4),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(2*n, 4*n, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv1d(4*n, 4*n, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv1d(4*n, 4*n, 9, stride=1, padding=4),
            nn.ReLU())

        self.amp5 = nn.AdaptiveMaxPool1d(5)
        
        self.linear = nn.Sequential(
            nn.Linear(20*n, 2*n),
            nn.ReLU(),
            nn.Linear(2*n, 1),
            nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        out = self.conv0(x)
        out = self.mp(out)
        out = self.conv1(out)
        out = self.mp(out)
        out = self.conv2(out)
        #print(out.size())
        out = self.amp5(out)
        out = out.view(out.shape[0], -1)
        #print(out.size())
        out = self.linear(out)

        return out



'''
x = torch.randn(1,12,10039)
net = ModelPlain(n=16)
y = net(x)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('This model has', total_params, 'parameters.')
print(y)'''


