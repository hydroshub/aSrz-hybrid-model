import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import date
import config  # External configuration


# Fully connected feed-forward network for generating static parameters from attributes
class StaticParamGenerator(nn.Module):
    def __init__(self, input_dim, output_keys, hidden_dim=16, dropout=0.3):
        super().__init__()
        self.output_keys = output_keys
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(output_keys))
        )

    def forward(self, static_input):
        raw_out = self.network(static_input)
        return dict(zip(self.output_keys, torch.split(raw_out, 1, dim=-1)))


class HydroProcess(nn.Module):
    def __init__(self):
        super().__init__()

    def _piecewise(self, soil_moisture, wetpoint, epsilon=0.03):
        left = soil_moisture / wetpoint
        right = epsilon * (soil_moisture - wetpoint) + 1.0
        response = torch.where(soil_moisture <= wetpoint, left, right)
        
        return torch.clamp(response, min=0.0, max=1.0)

    def rainsnow_partition(self, forcings, params, states, fluxes):
        # Partition precipitation into snow and rain based on temperature
        t, p = forcings['T'], forcings['P']
        tsnow = params['snow_tsnow']
        train = params['snow_train']

        snow_frac = torch.where(t <= tsnow, 1.0,
                                torch.where(t >= train, 0.0,
                                            (train - t) / (train - tsnow)))

        fluxes['snow'] = p * snow_frac
        fluxes['rain'] = p - fluxes['snow']
        return forcings, params, states, fluxes

    def snow_bucket(self, forcings, params, states, fluxes):
        # Update snowpack by first adding snowfall, then melting
        t = forcings['T']
        fmt = params['snow_fmt']
        tmelt = 0.0

        # Add snowfall to snow storage first
        s_tmp = states['S_SNOW'] + fluxes['snow']

        # Compute potential melt
        potential_melt = torch.where(t > tmelt, (t - tmelt) * fmt, torch.zeros_like(t))

        # Limit melt to available snow
        melt = torch.min(s_tmp, potential_melt)

        # Update state and flux
        fluxes['melt'] = melt
        states['S_SNOW'] = s_tmp - melt
        return forcings, params, states, fluxes

    def partition_available_water(self, forcings, params, states, fluxes):
        # Partition water into:
        # - to_avai: fraction available to ecosystem functioning
        # - to_unavai: remainder bypassing to runoff
        total = fluxes['rain'] + fluxes['melt']
        fluxes['to_avai'] = total * params['split_avai']
        fluxes['to_unavai'] = total * params['split_unavai']
        return forcings, params, states, fluxes

    def avai_bucket(self, forcings, params, states, fluxes):
        # Update ecosystem-available soil water bucket
        s_tmp = states['S_AVAI'] + fluxes['to_avai']

        # Relative soil moisture (0–1) using fixed capacity threshold
        relmoist = torch.clamp(s_tmp / (params['avai_cap'] + 1e-6), 0.0, 1.0)
        # states['f_wetness'] = self._piecewise(relmoist, params['avai_wetpoint'])
        sigmoid_center = params['avai_wetpoint99'] - (1.0 / params['avai_wetpoint_k']) * 4.59511985013 # ln99 = 4.59511985013
        states['f_wetness'] = 1.0 / (1.0 + torch.exp(-params['avai_wetpoint_k'] * (relmoist - sigmoid_center)))
        
        # Vegetation control via LAI
        f_phenology = 1 - params['avai_beta'] * (1 - forcings['LAI_normed']) ** 2
        states['f_phenology'] = f_phenology

        # Radiation-driven potential evapotranspiration (converted from MJ/m2 to mm)
        rad_evap_demand = torch.clamp(forcings['RAD'] * 0.0864 / 2.45, min=0.0)
        fluxes['rad_evap_demand'] = rad_evap_demand

        # Actual ET limited by moisture and vegetation
        et = rad_evap_demand * params['avai_efmax'] * states['f_wetness'] * states['f_phenology']
        fluxes['et'] = torch.minimum(et, s_tmp)

        # Update state by subtracting ET
        states['S_AVAI'] = torch.clamp(s_tmp - fluxes['et'], min=0.0)
        states['rel_moist'] = torch.clamp(states['S_AVAI'] / (params['avai_cap'] + 1e-6), min=0.0, max=1.0)
        
        return forcings, params, states, fluxes

    def fast_bucket(self, forcings, params, states, fluxes):
        # Simulate fast flow bucket
        s_tmp = states['S_FAST'] + fluxes['to_unavai']

        # Quick flow response (e.g., surface runoff or rapid lateral flow)
        q_fast = params['fast_kf'] * s_tmp
        fluxes['q_fast'] = torch.clamp(q_fast, max=s_tmp)

        # Percolation to slower bucket
        fluxes['perc'] = torch.minimum(params['fast_perc'], s_tmp - fluxes['q_fast'])

        # Update fast bucket storage
        states['S_FAST'] = torch.clamp(s_tmp - fluxes['q_fast'] - fluxes['perc'], min=0.0)
        return forcings, params, states, fluxes

    
    def slow_bucket(self, forcings, params, states, fluxes):
        # Simulate slow flow bucket (e.g., groundwater or baseflow reservoir)
        s_tmp = states['S_SLOW'] + fluxes['perc']

        # Outflow from slow reservoir
        q_slow = params['slow_ks'] * s_tmp
        fluxes['q_slow'] = torch.clamp(q_slow, max=s_tmp)

        # Update slow bucket storage
        states['S_SLOW'] = s_tmp - fluxes['q_slow']
        return forcings, params, states, fluxes

    def river_bucket(self, forcings, params, states, fluxes):
        q_total = fluxes['q_fast'] + fluxes['q_slow']  # [B, T, 1]
        B, T, _ = q_total.shape
    
        # Use first timestep for river_maxbas (static per basin)
        river_maxbas = params['river_maxbas'][:, 0, 0]  # [B]
    
        maxbas = river_maxbas.clamp(min=1.0).ceil().long()  # [B]
        max_len = maxbas.max().item()
        if max_len % 2 == 0:
            max_len += 1
        half = (max_len - 1) // 2
    
        q_pad = F.pad(q_total.transpose(1, 2), (half, half), mode='replicate')  # [B, 1, T + 2*half]
        q_unfold = q_pad.unfold(dimension=2, size=max_len, step=1).squeeze(1)   # [B, T, max_len]
    
        kernel = torch.arange(-half, half + 1, device=q_total.device).float()
        kernel_base = 1.0 - torch.abs(kernel) / (half + 1)
        kernel_base = kernel_base / kernel_base.sum()
        kernel_all = kernel_base.repeat(B, 1)  # [B, max_len]
    
        fluxes['q_routed'] = torch.einsum('btk,bk->bt', q_unfold, kernel_all).unsqueeze(-1)  # [B, T, 1]
    
        return forcings, params, states, fluxes

def check_nan(params, states, step, mode):
    for k, v in params.items():
        if torch.isnan(v).any():
            print(f'NAN in param {k} at step {step}, mode={mode}')
    for k, v in states.items():
        if torch.isnan(v).any():
            print(f'NAN in state {k} at step {step}, mode={mode}')

class HybridModel(nn.Module):
    def __init__(self, param_range, attr_cols):
        super().__init__()

        self.param_range = param_range
        self.attr_cols = attr_cols
        self.attr_dim_nonveg = len(attr_cols) - 5
        self.attr_dim_veg_static = len(attr_cols) + 1

        # Group 1: Non-vegetation params
        self.param_generators_nonveg = nn.ModuleDict({
            'snow_tsnow': StaticParamGenerator(self.attr_dim_nonveg, ['snow_tsnow']),
            'snow_train': StaticParamGenerator(self.attr_dim_nonveg, ['snow_train']),
            'snow_fmt': StaticParamGenerator(self.attr_dim_nonveg, ['snow_fmt']),
            'fast_kf': StaticParamGenerator(self.attr_dim_nonveg, ['fast_kf']),
            'fast_perc': StaticParamGenerator(self.attr_dim_nonveg, ['fast_perc']),
            'slow_ks': StaticParamGenerator(self.attr_dim_nonveg, ['slow_ks']),
            'river_maxbas': StaticParamGenerator(self.attr_dim_nonveg, ['river_maxbas']),

        })
        
        # Group 2: Vegetation params
        self.param_generators_veg_static = nn.ModuleDict({
            'split_k': StaticParamGenerator(self.attr_dim_veg_static, ['split_k']),
            'avai_beta': StaticParamGenerator(self.attr_dim_veg_static, ['avai_beta']),
            'avai_efmax': StaticParamGenerator(self.attr_dim_veg_static, ['avai_efmax']),
            'avai_cap_base': StaticParamGenerator(self.attr_dim_veg_static, ['avai_cap_base']),
            'avai_wetpoint99': StaticParamGenerator(self.attr_dim_veg_static, ['avai_wetpoint99']),
        })

        self.process = HydroProcess()

    
    def init_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        states = {
            'S_SNOW': torch.zeros(batch_size, 1, device=device),
            'S_AVAI': torch.zeros(batch_size, 1, device=device),
            'S_FAST': torch.zeros(batch_size, 1, device=device),
            'S_SLOW': torch.zeros(batch_size, 1, device=device),
            'rel_moist': torch.zeros(batch_size, 1, device=device),
            'f_wetness': torch.zeros(batch_size, 1, device=device),
        }
        return states

    def generate_params(self, attributes, forcings, states):
    
        param_raw = {}
        
        # Non-vegetation → static_other
        for k, gen in self.param_generators_nonveg.items():
            param_raw[k] = gen(attributes['static_other'])[k]
        
        # Vegetation static → static_veg
        for k, gen in self.param_generators_veg_static.items():
            param_raw[k] = gen(attributes['static_veg'])[k]
        
        params = {}
        for k, v in param_raw.items():
            low, high = self.param_range[k]
            v_scaled = torch.sigmoid(v) * (high - low) + low
            params[k] = torch.clamp(v_scaled, min=low, max=high)

        
        rel_moist = torch.clamp(states['rel_moist'], min=0.01, max=0.99)

        #split_avai = params['split_b'] * ((1.0 - rel_moist) ** params['split_k'])
        split_avai = (1.0 - rel_moist) ** params['split_k']
        params['split_avai'] = 0.05 + 0.95 * split_avai
        params['split_unavai'] = 1.0 - params['split_avai']

        lai_scaled   = forcings['LAI_avg'] / (forcings['LAI_avg'] + 3)
        slope_scaled = 1 - (torch.sigmoid(forcings['SLOPE_avg']) - 0.5)
        params['avai_cap'] = params['avai_cap_base'] * lai_scaled * (1 + slope_scaled)
        
        params['avai_wetpoint_k'] = 2 * 4.59511985013 / params['avai_wetpoint99'] # ln99 = 4.59511985013
        params['avai_slope'] = params['avai_efmax'] / params['avai_wetpoint99']
                
        return params
    

    def forward(self, x_seq, date_seq, mode='train', show_progress=False, states=None, log_to_cpu=False):
        
        B, T, _ = x_seq.shape
        device = x_seq.device
    
        # ---- ----- ----
        # Initialize logs
        # ---- ----- ----
        all_fluxes, all_states, all_params = {}, {}, {}
        all_tws, all_date = [], []
    
        if date_seq.ndim == 1:
            date_seq = date_seq.unsqueeze(0).repeat(B, 1)
    
        # ---- ----- ----
        # Initialize states
        # ---- ----- ----
        if states is None:
            states = self.init_states(B, device=device)
        
        # ---- ----- ----
        # Prepare normed LAI
        # ---- ----- ----
        # Extract and normalize LAI
        lai_raw = x_seq[:, :, 4:5]  # [B, T, 1]
        lai_avg = lai_raw.mean(dim=1)  # [B, 1]

        # Local LAI_norm
        lai_max = lai_raw.quantile(0.99, dim=1, keepdim=True)
        lai_min = lai_raw.quantile(0.01, dim=1, keepdim=True)
        lai_normed_seq = torch.clamp((lai_raw - lai_min) / (lai_max - lai_min + 1e-6), 0.0, 1.0)

        # Extract and normalize terrain slope
        slope_idx = self.attr_cols.index('slp_dg_sav')
        slope_raw = x_seq[:, :, 5+slope_idx:6+slope_idx]  # [B, T, 1] the first 5 are meteo and LAI
        slope_avg = slope_raw.mean(dim=1)  # [B, 1]

        # ---- ----- ----
        # Prepare static inputs
        # ---- ----- ----
        # Extract and normalize Temperature
        temp_raw = x_seq[:, :, 1:2]  # [B, T, 1]
        temp_max = temp_raw.quantile(0.99, dim=1, keepdim=True)
        temp_min = temp_raw.quantile(0.01, dim=1, keepdim=True)
        temp_normed_seq = torch.clamp((temp_raw - temp_min) / (temp_max - temp_min + 1e-6), 0.0, 1.0)
        
        # Land cover fractions
        land_cover = {
            'forest': x_seq[:, :, 5:6].mean(dim=1) / 100.0,
            'shrub': x_seq[:, :, 6:7].mean(dim=1) / 100.0,
            'grass': x_seq[:, :, 7:8].mean(dim=1) / 100.0,
            'crop': x_seq[:, :, 8:9].mean(dim=1) / 100.0,
            'others': x_seq[:, :, 9:10].mean(dim=1) / 100.0,
        }

        # LAI × land cover
        #lai_components = [lai_avg * frac for frac in land_cover.values()] 
        veg_frac = torch.cat([frac for frac in land_cover.values()], dim=-1)
        #static_lai = torch.cat(lai_components, dim=-1)
    
        # Static other attributes
        static_other = x_seq[:, :, 10:].mean(dim=1) 
    
        # Static veg
        static_veg = torch.cat([lai_avg, veg_frac, static_other], dim=-1)
    
        # ---- ----- ----
        # Time loop
        # ---- ----- ----
        time_iter = range(T)

        if show_progress:
            from tqdm import tqdm
            time_iter = tqdm(time_iter, desc="Forward", ncols=100)

        # Loop over timesteps
        for t in time_iter:
            # ---- forcings inputs ----
            forcings = {
                'P': x_seq[:, t, 0:1],
                'T': x_seq[:, t, 1:2],
                'RAD': x_seq[:, t, 2:3] + x_seq[:, t, 3:4],
                'LAI': x_seq[:, t, 4:5],
                'LAI_avg': lai_avg,
                'SLOPE_avg': slope_avg,
                'LAI_normed': lai_normed_seq[:, t, :],
                'T_normed': temp_normed_seq[:, t, :],
            }
            
            # ---- Assemble attributes ----
            attributes = {
                'static_other': static_other,          # for non-veg params
                'static_veg': static_veg,              # for veg params (static)
            }        

            # ---- Generate parameters ----
            params = self.generate_params(attributes, forcings, states)

            # ---- Run hydro process ----
            fluxes = {}
            for fn in [
                self.process.rainsnow_partition,
                self.process.snow_bucket,
                self.process.partition_available_water,
                self.process.avai_bucket,
                self.process.fast_bucket,
                self.process.slow_bucket,
            ]:
                forcings, params, states, fluxes = fn(forcings, params, states, fluxes)

            # ---- Update logs ----
            states['S_TWS'] = states['S_SNOW'] + states['S_AVAI'] + states['S_FAST'] + states['S_SLOW']
            all_tws.append(states['S_TWS'].unsqueeze(1))
            all_date.append(date_seq[:, t].unsqueeze(1))

            # Record fluxes, states, parameters
            for k, v in fluxes.items():
                v_log = v.cpu().detach() if log_to_cpu else v
                all_fluxes.setdefault(k, []).append(v_log.unsqueeze(1))
    
            for k, v in states.items():
                v_log = v.cpu().detach() if log_to_cpu else v
                all_states.setdefault(k, []).append(v_log.unsqueeze(1))
    
            for k, v in params.items():
                v_log = v.cpu().detach() if log_to_cpu else v
                all_params.setdefault(k, []).append(v_log.unsqueeze(1))


            # print('avai_cap', params['avai_cap'].min().item(), params['avai_cap'].max().item())

            # check nan
            # check_nan(params, states, t, mode)

        # ---- ----- ----
        # Final post-processing
        # ---- ----- ----
        all_fluxes = {k: torch.cat(v, dim=1) for k, v in all_fluxes.items()}
        all_states = {k: torch.cat(v, dim=1) for k, v in all_states.items()}
        all_params = {k: torch.cat(v, dim=1) for k, v in all_params.items()}

        # ---- ----- ----
        # Run river_bucket on the full sequence
        # ---- ----- ----
        _, _, _, all_fluxes = self.process.river_bucket(
            forcings, all_params, all_states, all_fluxes
        )

        # ---- ----- ----
        # Aggregate TWSA for monthly scale
        # ---- ----- ----        
        # ---- Compute monthly TWS and anomaly ---- 
        all_tws_tensor = torch.cat(all_tws, dim=1)  # [B, T, 1]
        all_date_tensor = torch.cat(all_date, dim=1)  # [B, T, 1]
        date_list = [date.fromordinal(int(d)) for d in all_date_tensor[0].cpu().numpy()]

        month_starts = [i for i in range(len(date_list)) if i == 0 or (date_list[i].month != date_list[i - 1].month)]
        month_starts.append(len(date_list))
        monthly_tws = [torch.nanmean(all_tws_tensor[:, s:e, :], dim=1) for s, e in zip(month_starts[:-1], month_starts[1:])]
        monthly_tensor = torch.stack(monthly_tws, dim=1)  # [B, M, 1]

        # ---- GRACE baseline reference: Jan 2004 – Dec 2009 ---- 
        month_dates = [date_list[i] for i in month_starts[:-1]]
        baseline_mask = [(d >= date(2004, 1, 1)) and (d <= date(2009, 12, 31)) for d in month_dates]
        baseline_mask = torch.tensor(baseline_mask, device=monthly_tensor.device, dtype=torch.bool)
        baseline_idx = baseline_mask.nonzero(as_tuple=False).squeeze(-1)

        baseline = monthly_tensor[:, baseline_idx, :].nanmean(dim=1, keepdim=True)
        anomaly_tensor = monthly_tensor - baseline

        all_states['TWS_monthly'] = monthly_tensor
        all_states['TWSA_monthly'] = anomaly_tensor

        # ---- ----- ----
        # Output selection
        # ---- ----- ----
        if mode == 'train':
            return {
                'q': all_fluxes['q_routed'],
                'et': all_fluxes['et'],
                'swe': all_states['S_SNOW'],
                'twsa_anomaly': all_states['TWSA_monthly'],
            }
        elif mode == 'full':
            return {
                'fluxes': all_fluxes,
                'states': all_states,
                'params': all_params,
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")


    
