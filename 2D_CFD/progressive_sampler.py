class ProgressiveSampleScheduler:
    def __init__(self, 
                 initial_samples=5, 
                 max_samples=50, 
                 increase_factor=1.5, 
                 patience=10, 
                 min_delta=0.001,
                 increase_mode='patience'):
        self.initial_samples = initial_samples
        self.max_samples = max_samples
        self.increase_factor = increase_factor
        self.patience = patience
        self.min_delta = min_delta
        self.increase_mode = increase_mode
        
        self.current_samples = initial_samples
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch_counter = 0
        self.sample_history = []
        
    def step(self, val_loss, epoch=None, val_loss_updated=True):
        samples_increased = False
        
        if epoch is not None:
            self.epoch_counter = epoch
        else:
            self.epoch_counter += 1
            
        if self.increase_mode == 'patience':
            samples_increased = self._update_patience_based(val_loss, val_loss_updated)
        elif self.increase_mode == 'epoch':
            samples_increased = self._update_epoch_based()
        else:
            raise ValueError(f"Unknown increase_mode: {self.increase_mode}")
        
        self.sample_history.append({
            'epoch': self.epoch_counter,
            'samples': self.current_samples,
            'val_loss': val_loss,
            'val_loss_updated': val_loss_updated,
            'increased': samples_increased
        })
        
        return self.current_samples, samples_increased
    
    def _update_patience_based(self, val_loss, val_loss_updated=True):
        samples_increased = False
        
        if val_loss_updated:
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        else:
            self.patience_counter += 1
            
        if (self.patience_counter >= self.patience and 
            self.current_samples < self.max_samples):
            
            new_samples = min(
                int(self.current_samples * self.increase_factor),
                self.max_samples
            )
            
            if new_samples > self.current_samples:
                print(f"Increasing samples from {self.current_samples} to {new_samples} "
                      f"(patience: {self.patience_counter}, val_loss_updated: {val_loss_updated})")
                self.current_samples = new_samples
                samples_increased = True
                
                self.patience_counter = 0
        
        return samples_increased
    
    def _update_epoch_based(self):
        samples_increased = False
        
        milestone_epochs = [20, 40, 60, 80]
        
        if (self.epoch_counter in milestone_epochs and 
            self.current_samples < self.max_samples):
            
            new_samples = min(
                int(self.current_samples * self.increase_factor),
                self.max_samples
            )
            
            if new_samples > self.current_samples:
                print(f"Increasing samples from {self.current_samples} to {new_samples} "
                      f"(epoch milestone: {self.epoch_counter})")
                self.current_samples = new_samples
                samples_increased = True
        
        return samples_increased
    
    def get_current_samples(self):
        return self.current_samples
    
    def get_history(self):
        return self.sample_history
    
    def reset(self):
        self.current_samples = self.initial_samples
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch_counter = 0
        self.sample_history = []
    
    def state_dict(self):
        return {
            'current_samples': self.current_samples,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'epoch_counter': self.epoch_counter,
            'sample_history': self.sample_history
        }
    
    def load_state_dict(self, state_dict):
        self.current_samples = state_dict['current_samples']
        self.best_val_loss = state_dict['best_val_loss']
        self.patience_counter = state_dict['patience_counter']
        self.epoch_counter = state_dict['epoch_counter']
        self.sample_history = state_dict['sample_history']
