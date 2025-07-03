import torch
from torch.optim import AdamW


class Trainer:
    def __init__(self, model, train_loader, val_loader, configTrain, configLora, tokenizer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.configTrain = configTrain
        self.configLora = configLora
        self.tokenizer = tokenizer
        self.device = torch.device("xpu" if configTrain.get('device', 'auto') == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=configTrain['learning_rate'])

    def train(self):
        epochs = self.configTrain['epochs']
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} — Train Loss: {avg_loss:.4f}")
            self.validate(epoch)
        if self.configTrain.get('save_model', True):
            self.save_model()

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {epoch+1} — Val Loss: {avg_loss:.4f}")

    def save_model(self):
        output_dir = self.configTrain.get('output_dir', 'outputs/lora_finetuned')
        self.model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        print(f"\n💾 Model saved to {output_dir}")