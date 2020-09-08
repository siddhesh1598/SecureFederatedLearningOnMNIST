# import
import torch
import torch.optim as optim

import syft as sy

from utils import *
from model import Net

# arguments
class Arguments():

    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.001
        self.momentum = 0.5
        self.no_cuda = False
        self.log_interval = 50
        self.save_model = False

args = Arguments()


# update params
def updateParams():
	# get bob's params
    bob_params = list()
    for paramsB in list(model_bob.parameters()):
        bob_params.append(paramsB.fix_prec().share(bob, alice, jon).get())
    
    # get alice's params
    alice_params = list()
    for paramsA in list(model_alice.parameters()):
        alice_params.append(paramsA.fix_prec().share(bob, alice, jon).get())

    # get updated params
    updated_params = list()
    for (paramA, paramB) in zip(alice_params, bob_params):
        updated_params.append((paramA + paramB).get().float_precision() / 2)
    
    # update params
    with torch.no_grad():

        # alice's model
        model_alice.get()
        for m_params, up_params in zip(model_alice.parameters(), updated_params):
            m_params.set_(up_params)
        model_alice.send(alice)

        # bob's model
        model_bob.get()
        for m_params, up_params in zip(model_bob.parameters(), updated_params):
            m_params.set_(up_params)
        model_bob.send(bob)


# train
def train(args, model, device, train_loader, optimizer, epochs):

	# copy models and send it to workers
	model_bob = model.copy().send(bob)
	model_alice = model.copy().send(alice)

	model_bob.train()
	model_alice.train()

	# initialize optimizers for the models
	opt_bob = optim.SGD(model_bob.parameters(), lr=args.lr)
	opt_alice = optim.SGD(model_alice.parameters(), lr=args.lr)

	getModelOpt = {
	    "bob": (model_bob, opt_bob), 
	    "alice": (model_alice, opt_alice)
	}

	workers = [alice, bob]

	for epoch in range(epochs):

	    for batch_idx, batch in enumerate(train_loader): 
	    # batch: a dictionary containing a batch for each individual worker

	        for worker in workers:
	            _data, _target = batch[worker][0], batch[worker][1]

	            (_model, _opt) = getModelOpt[worker.id]
	            
	            _opt.zero_grad()
	            output = _model(_data)
	            loss = F.nll_loss(output, _target)
	            loss.backward()
	            _opt.step()
	        
	        if batch_idx % args.log_interval == 0:
	        	# display the loss and update the parameters of the models
	        	# after every interval

	            loss = loss.get()
	            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	                epoch + 1, batch_idx * args.batch_size, 
	                len(train_loader) * args.batch_size,
	                100. * batch_idx / len(train_loader), loss.item()))
	            
	            updateParams(model_alice, model_bob)

	return (model_alice, model_bob)


# aggregate
def aggregate(model, model_alice, model_bob, secure_worker=jon):

    model_alice.move(jon)
    model_bob.move(jon)

    for (_, paramA), (_, paramB), (_, param) in zip(model_alice.named_parameters(), 
                                            model_bob.named_parameters(),
                                            model.named_parameters()):
        param.data = (((paramA.data + paramB.data) / 2).get())

    return model


# test
def test(args, model, device, test_loader):

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():

        for _data, _target in test_loader:

            _data, _target = _data.to(device), _target.to(device)
            
            output = model(_data)
            test_loss += F.nll_loss(output, _target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(_target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

	# initialize device
	device = torch.device("cpu" if args.no_cuda else "cuda")    

	# create hook
	hook = sy.TorchHook(torch)

	# initialize workers
	bob = sy.VirtualWorker(hook, id="bob")  
	alice = sy.VirtualWorker(hook, id="alice")
	jon = sy.VirtualWorker(hook, id="jon")	# secure worker

	# get MNIST dataset loader
	train_loader = getTrainDataLoader((bob, alice), 
	    args.batch_size)
	test_loader = getTestDataLoader(args.test_batch_size)

	# initialize model and optimizer
	model = Net()
	optimizer = optim.SGD(model.parameters(), lr=args.lr)

	# train
	modelA, modelB = train(args, model, device, train_loader, optimizer, args.epochs)
	# aggregate
	model = aggregate(model, modelA, modelB, 
		secure_worker=jon)
	# test
	test(args, model, device, test_loader)

	# save model
	if (args.save_model):
    	torch.save(model.state_dict(), "fl_mnist_cnn.pt")

if __name__ == "__main__":
    main()