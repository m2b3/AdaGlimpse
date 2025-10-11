import torch
from torch import Tensor
from AdaGlimpse.architectures.rl.shared_memory import SharedMemory
from jaxtyping import Float


class RLStateReplaceHook:
    def __init__(self, avg_patch=None, avg_coords=None, avg_importance=None, avg_latent=None):
        self.avg_patch = avg_patch
        self.avg_coords = avg_coords
        self.avg_importance = avg_importance
        self.avg_latent = avg_latent

    def __call__(self, env_state: SharedMemory, out, score, attention, observation, next_state):
        if self.avg_patch is not None:
            next_state['patches'][:, :] = self.avg_patch.unsqueeze(0).unsqueeze(0)
        if self.avg_coords is not None:
            next_state['coords'][:, :] = self.avg_coords.unsqueeze(0).unsqueeze(0)
        if self.avg_importance is not None:
            next_state['attention'][:, :] = self.avg_importance.unsqueeze(0).unsqueeze(0)
        if self.avg_latent is not None:
            next_state['observation'][:, :] = self.avg_latent.unsqueeze(0).unsqueeze(0)


class RLUserHook:
    def __init__(self, avg_latent=False):
        self.avg_latent = avg_latent

        self.images = []
        self.latent = []
        self.out = []
        self.scores = []
        self.coords = []
        self.patches = []
        self.current_out = []
        self.current_scores = []
        self.targets = []
        self.done = []
        self.current_done = []
        self.importance = []
        self.current_importance = []
        self.latent = None
        self.batch_glimpse_timings = []
        self.current_batch_glimpse_timings = []
        self.current_time = 0.0

    def __call__(self, env_state: SharedMemory, out, score, attention, observation, next_state, batched_glimpse_dur: float):

        # This function is called once per environment step (i.e., once per glimpse)
        self.current_time += batched_glimpse_dur
        self.current_batch_glimpse_timings.append(self.current_time)
        self.current_out.append(out.clone().detach().cpu())
        self.current_done.append(env_state.done.clone().detach().cpu())
        """self.current_scores.append(score.clone().detach().cpu())
        self.current_done.append(env_state.done.clone().detach().cpu())
        self.current_importance.append(attention.clone().detach().cpu())"""

        if env_state.is_done:
            # Enter this if statement after the last glimpse of the BATCH
            """self.images.append(env_state.images.clone().detach().cpu())
            self.coords.append(env_state.current_coords.clone().detach().cpu())
            self.patches.append(env_state.current_patches.clone().detach().cpu())
            self.out.append(torch.stack(self.current_out, dim=1))
            self.current_out = []
            self.scores.append(torch.stack(self.current_scores, dim=1))
            self.current_scores = []
            self.targets.append(env_state.target.clone().detach().cpu())
            self.done.append(torch.stack(self.current_done, dim=1))
            self.current_done = []
            self.importance.append(torch.stack(self.current_importance, dim=1))
            self.current_importance = []"""
            self.out.append(torch.stack(self.current_out, dim=1))
            self.current_out = []
            self.done.append(torch.stack(self.current_done, dim=1))
            self.current_done = []
            self.targets.append(env_state.target.clone().detach().cpu())

            batch_timing_tensor = torch.tensor(self.current_batch_glimpse_timings, dtype=torch.float32)
            self.batch_glimpse_timings.append(batch_timing_tensor)
            self.current_batch_glimpse_timings = []  # Reset for next batch
            self.current_time = 0.0
                
            """if self.avg_latent:
                if self.latent is None:
                    self.latent = observation.clone().detach().cpu().mean(dim=0).mean(dim=0)
                else:
                    self.latent += observation.clone().detach().cpu().mean(dim=0).mean(dim=0)"""

    def compute(self):
        glimpse_cum_timings: Float[Tensor, 'num_batches, num_glimpses'] = torch.stack(self.batch_glimpse_timings, dim=0)

        result = {
            # "images": torch.cat(self.images, dim=0),
            "out": torch.cat(self.out, dim=0),
            "glimpse_cum_timings": glimpse_cum_timings,
            # "scores": torch.cat(self.scores, dim=0),
            # "coords": torch.cat(self.coords, dim=0),
            # "patches": torch.cat(self.patches, dim=0),
            "targets": torch.cat(self.targets, dim=0),
            "done": torch.cat(self.done, dim=0),
            # "importance": torch.cat(self.importance, dim=0),
            # "latent": self.latent / len(self.done) if self.latent is not None else None
        }
   
        return result
