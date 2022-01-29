import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

from vae import *
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# Reconstruction + β * KL divergence losses summed over all elements and batch
'''
Assume p, q are Normal distributions, the KL term looks like this:
'''
def loss_function_normal(x_hat, x, mu, logvar, β=0.01):
  BCE = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
  KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
  return BCE + β * KLD


def calc_params(x_hat, x, mu, logvar):
  # 1. define the first two probabilities (in this case Normal for both)
  std = torch.exp(logvar / 2)
  q = torch.distributions.Normal(mu, std)
  z = q.rsample()
  p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))

  # 2. get the probabilities from the equation
  log_qzx = q.log_prob(z)
  log_pz = p.log_prob(z)
  logpx_z = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

  return log_qzx, log_pz, logpx_z


def loss_function_ELBO(x_hat, x, mu, logvar):
  log_qzx, log_pz, logpx_z = calc_params(x_hat, x, mu, logvar)
  kld_mc = torch.mean(log_qzx) - torch.mean(log_pz)
  elbo = torch.mean(logpx_z) - kld_mc
  return elbo


def loss_function_Renyi(x_hat, x, mu, logvar, alpha=1):
  # alpha is close to 1
  if np.abs(alpha - 1.0) < 10e-3:
    return loss_function_ELBO(x_hat, x, mu, logvar)

  log_qzx, log_pz, logpx_z = calc_params(x_hat, x, mu, logvar)
  ratio = (alpha-1) * (log_pz + logpx_z - log_qzx)
  const = torch.max(ratio)
  expectant = torch.exp((ratio - const))

  renyi_div = (torch.log(torch.mean(expectant)) + const) / (alpha-1)
  return renyi_div

#
# def loss_function_Renyi(x_hat, x, mu, logvar, alpha=1):
#
#   log_qzx, log_pz, logpx_z = calc_params(x_hat, x, mu, logvar)
#   renyi_div = RenyiDiv(log_qzx.detach().numpy(), log_pz.detach().numpy(), alpha)
#   res = torch.mean(logpx_z).item() - renyi_div
#   return torch(res)
#

def kl_divergence(p, q):
  return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def RenyiDiv(p,q,alpha=1):
  if (alpha==1):
    return kl_divergence(p, q)

  if (alpha==0):
    return -np.log(np.sum(np.where(p != 0, q, 0)))

  return 1.0/(alpha-1) * np.log(np.sum(np.where(p != 0, p * np.power(p / q, alpha-1), 0)))


def calc_log_px(centers, stds, num_samples, x, y):
  # The probability to be in this gaussian * the probability of this gaussian
  index = y.item()
  rvd = multivariate_normal(centers[index], stds[index])
  sample = x.reshape(x.shape[-1]*x.shape[-2]).numpy()
  prob = rvd.pdf(sample)

  print("center = ", centers[index][:5], "   sample = ", sample[:5], "   probability = ", prob)

  px = (num_samples[index] / np.sum(num_samples)) * prob
  if px == 0.0:
      px = px + 1e-10
  return np.log(px)


# Training and testing the VAE
def train(train_loader, test_loader, alpha, num_epochs=20, learning_rate=1e-10):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = VAE().to(device)
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
  scheduler = ExponentialLR(optimizer, gamma=0.9)
  scheduler2 = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

  # Calculate log_px outside the loop since this value is constant
  codes = {'lowerbound': [], 'recon_loss': []}
  for epoch in range(num_epochs):
    # Training
    net.train()
    train_loss = 0
    for x, y in train_loader:
      x = x.to(device)

      # ===================forward=====================
      x_hat, mu, logvar = net(x)
      loss = loss_function_Renyi(x_hat, x, mu, logvar, alpha)
      train_loss += loss.item()

      # ===================backward====================
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_value_(net.parameters(), clip_value=1.0)

      optimizer.step()

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

    # Testing
    with torch.no_grad():
      net.eval()
      test_loss = 0
      for x, y in test_loader:
        x = x.to(device)

        # ===================forward=====================
        x_hat, mu, logvar = net(x)
        test_loss += loss_function_Renyi(x_hat, x, mu, logvar, alpha).item()

        # ===================log========================
        # reconstruction loss
        recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        rec_loss_val = recon_loss.mean()

      codes['recon_loss'].append(rec_loss_val.item())

      test_loss /= len(test_loader.dataset)
      print(f'====> Test set loss: {test_loss:.4f}')
      codes['lowerbound'].append(test_loss)

    scheduler.step()
    scheduler2.step()

  return net, codes


def plot_learning_curve(num_epochs, alpha_values):
    # evenly sampled time at 200ms intervals
    epochs = np.arange(num_epochs)
    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()
    # ax.set_ylim([0, 500])
    ax.set_ylim([-600, 600])


    for alpha in alpha_values:
      learning_curve_alpha = np.loadtxt('learning_curves_alpha_{}.csv'.format(alpha))
      vals = learning_curve_alpha[:num_epochs]
      # if alpha > 0:
      vals = [-val for val in vals]
      ax.plot(epochs, vals, label="Renyi, alpha={}".format(alpha))

    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-small')

    plt.savefig('learning_curves.png')

