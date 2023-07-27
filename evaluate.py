import torch
from sklearn.metrics import fbeta_score
import torch.distributed


from utils.logger import get_logger, DEBUG
from utils.dist import is_distributed

logger = get_logger('eval')


def evaluate(model, criterion, eval_loader, epochs, device, epoch):
    model.eval()
    eval_epoch_loss = torch.zeros(1, device=device, requires_grad=False)
    eval_epoch_fbeta = torch.zeros(1, device=device, requires_grad=False)

    percentages = [int(len(eval_loader) * i / 100) for i in range(0, 100, 20)][1:]
    with torch.no_grad():
        for i, (sub_volumes, ink_labels) in enumerate(eval_loader):
            outputs = model(sub_volumes.to(device))

            outputs = outputs.squeeze(1)
            loss = criterion(outputs, ink_labels.to(device).float())

            eval_epoch_loss += loss.item()

            outputs = torch.sigmoid(outputs.detach())
            ink_predictions = torch.where(outputs.detach() > 0.5, 1, 0).cpu()

            ink_labels = ink_labels.reshape(-1)
            ink_predictions = ink_predictions.reshape(-1)
            ink_labels = torch.where(ink_labels == 2, 0, ink_labels)
            ink_predictions = torch.where(ink_predictions == 2, 0, ink_predictions)

            eval_epoch_fbeta += fbeta_score(ink_labels.numpy(), ink_predictions.numpy(), beta=0.5, zero_division=0)

            if i in percentages:
                logger.info(f"Epoch [{epoch}]/[{epochs}]. Eval Step [{i}][{len(eval_loader)}]. "
                            f"Eval Loss: {eval_epoch_loss.item() / i}. Eval Fbeta: {eval_epoch_fbeta.item() / i}")

            if is_distributed():
                torch.distributed.barrier()

    return eval_epoch_loss / len(eval_loader), eval_epoch_fbeta / len(eval_loader)
