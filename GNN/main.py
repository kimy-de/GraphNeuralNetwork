from gnn import *
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GNN().to(device)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=trans), batch_size=32)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True, transform=trans), batch_size=32, shuffle=False)
    #
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(11):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(
            'Epoch: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(ep,
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))