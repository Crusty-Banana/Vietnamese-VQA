from tqdm.auto import tqdm
import torch
from torchmetrics import F1Score
import logging
import os

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, scheduler, device, run_dir, run_name):
    train_loss = []
    val_loss = []
    train_f1 = []
    val_f1 = []

    log_file = os.path.join(run_dir, f"training_log_{run_name}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    for epoch in range(num_epochs):
        model.train()
        running_train_loss, running_train_f1 = 0, 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [TRAIN]')
        for batch_dict in train_bar:
            # Move the inputs and targets to the device (CPU or GPU)
            input_ids = batch_dict['input_ids'].to(device)
            pixel_values = batch_dict['pixel_values'].to(device)
            attention_mask = batch_dict['attention_mask'].to(device)
            labels = batch_dict['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, pixel_values, attention_mask, labels=labels)
            loss = outputs['loss']

            f1 = F1Score(task="multiclass", num_classes=250027, top_k=1, ignore_index=-100).to(device)
            s = f1(outputs['logits'].argmax(dim=2), labels)

            running_train_loss += loss.item()
            running_train_f1 += s.clone().cpu().numpy()

            train_bar.set_postfix(loss=running_train_loss / (train_bar.n + 1),
                                  f1=running_train_f1 / (train_bar.n + 1))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_f1 = running_train_f1 / len(train_loader)
        train_loss.append(epoch_train_loss)
        train_f1.append(epoch_train_f1)

        # Use learning scheduler
        scheduler.step()

        # Test on validation dataset
        model.eval()
        running_val_loss, running_val_f1 = 0, 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [VAL]')
        with torch.no_grad():
            for batch_dict in val_bar:
                input_ids = batch_dict['input_ids'].to(device)
                pixel_values = batch_dict['pixel_values'].to(device)
                attention_mask = batch_dict['attention_mask'].to(device)
                labels = batch_dict['labels'].to(device)

                outputs = model(input_ids, pixel_values, attention_mask, labels=labels)

                lm_logits = outputs['logits']
                labels = labels.to(lm_logits.device)
                loss = criterion(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

                # Compute f1_score
                f1 = F1Score(task="multiclass", num_classes=250027, top_k=1, ignore_index=-100).to(device)
                s = f1(lm_logits.argmax(dim=2), labels)
                running_val_loss += loss.item()
                running_val_f1 += s.clone().cpu().numpy()

                # Update tqdm bar
                val_bar.set_postfix(loss=running_val_loss / (val_bar.n + 1),
                                    f1=running_val_f1 / (val_bar.n + 1))

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_f1 = running_val_f1 / len(val_loader)
        val_loss.append(epoch_val_loss)
        val_f1.append(epoch_val_f1)


        print("Epoch [{}/{}], Train Loss: {:.4f}, Train F1: {:.4f}, LearningRate: {} ".format(epoch+1, num_epochs, epoch_train_loss, epoch_train_f1, scheduler.get_last_lr()))
        print("Epoch [{}/{}], Val Loss: {:.4f}, Val F1: {:.4f}".format(epoch+1, num_epochs, epoch_val_loss, epoch_val_f1))
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train F1: {epoch_train_f1:.4f}, LearningRate: {scheduler.get_last_lr()} ")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {epoch_val_loss:.4f}, Val F1: {epoch_val_f1:.4f}")

    # Save the trained model within the created folder
    model_name = os.path.join(run_dir, f"model_{run_name}.pth")
    torch.save(model.state_dict(), model_name)

        # Save the trained model
        # if (epoch+1) > 0:
        #     torch.save(model.state_dict(), model_outputs + "/ViTmBART_epoch" + str(epoch+19) + ".pth")
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # model_name = f"model_{current_time}.pth"
    # torch.save(model.state_dict(), model_name)

    return model, train_loss, val_loss, train_f1, val_f1