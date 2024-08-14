import wandb
import torch
import numpy as np
from torch.autograd import Variable

# Initialize wandb
wandb.init(project="DCGAN-CelebA-Inpainting", config=opt.__dict__)

# Training loop
for epoch in range(opt.n_epochs):
    for i, (masked_imgs, real_imgs) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(masked_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(masked_imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        masked_imgs = Variable(masked_imgs.type(Tensor))
        real_imgs = Variable(real_imgs.type(Tensor))

        # Train Generator
        optimizer_G.zero_grad()

        # Sample noise as generator input (optional, based on specific architecture)
        z = Variable(Tensor(np.random.normal(0, 1, (masked_imgs.size(0), opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(masked_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid) + context_loss(gen_imgs, real_imgs)

        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Log losses
        wandb.log({"G_loss": g_loss.item(), "D_loss": d_loss.item(), "epoch": epoch})

        # Every few batches, save and log images
        if i % opt.sample_interval == 0:
            # Save generated images
            sample_images = gen_imgs.data[:16]
            wandb.log({"generated_images": [wandb.Image(sample_images, caption=f"Epoch {epoch}, Batch {i}")]})

            # Log model weights and biases
            wandb.log({
                # Generator Weights and Biases
                "Generator Weights (l1.0)": wandb.Histogram(generator.state_dict()['l1.0.weight'].cpu().numpy()),
                "Generator Biases (l1.0)": wandb.Histogram(generator.state_dict()['l1.0.bias'].cpu().numpy()),
                "Generator Weights (conv_blocks.0)": wandb.Histogram(generator.state_dict()['conv_blocks.0.weight'].cpu().numpy()),
                "Generator Biases (conv_blocks.0)": wandb.Histogram(generator.state_dict()['conv_blocks.0.bias'].cpu().numpy()),

                # Discriminator Weights and Biases
                "Discriminator Weights (model.0)": wandb.Histogram(discriminator.state_dict()['model.0.weight'].cpu().numpy()),
                "Discriminator Biases (model.0)": wandb.Histogram(discriminator.state_dict()['model.0.bias'].cpu().numpy()),
                "Discriminator Weights (model.3)": wandb.Histogram(discriminator.state_dict()['model.3.weight'].cpu().numpy()),
                "Discriminator Biases (model.3)": wandb.Histogram(discriminator.state_dict()['model.3.bias'].cpu().numpy()),
            })

# Finish wandb run
wandb.finish()

