import numpy as np

GRID_Z, GRID_X, GRID_Y = 10, 20, 20   # layers × x × y（中上と同じ軸順）
X_MIN, X_MAX = -700.0, 700.0
Y_MIN, Y_MAX = -700.0, 700.0

def build_sili_voxel(event):
  """Si(Li)ヒット → 3Dボクセルグリッド (10,20,20) = layers×x×y"""
  vids      = np.array(event['volume_id'])
  energies  = np.array(event['energy'])
  positions = np.array(event['positions'])

  layer_idx = (vids // 1000000).astype(np.int64)
  is_sili   = layer_idx >= 200
  if not is_sili.any():
      return np.zeros((GRID_Z, GRID_X, GRID_Y), dtype=np.float32)

  layer   = layer_idx[is_sili] % 100
  valid   = layer < GRID_Z
  pos_s   = positions[is_sili][valid]
  e_s     = energies[is_sili][valid]
  layer_v = layer[valid]

  if len(e_s) == 0:
      return np.zeros((GRID_Z, GRID_X, GRID_Y), dtype=np.float32)

  xi = ((pos_s[:, 0] - X_MIN) / (X_MAX - X_MIN) * GRID_X).astype(np.int32)
  yi = ((pos_s[:, 1] - Y_MIN) / (Y_MAX - Y_MIN) * GRID_Y).astype(np.int32)
  mask = (xi >= 0) & (xi < GRID_X) & (yi >= 0) & (yi < GRID_Y)

  grid = np.zeros((GRID_Z, GRID_X, GRID_Y), dtype=np.float32)
  np.add.at(grid, (layer_v[mask], xi[mask], yi[mask]), e_s[mask])
  return grid


def build_tof_features(event):
  """TOF関連11次元特徴量"""
  vids      = np.array(event['volume_id'])
  energies  = np.array(event['energy'])
  positions = np.array(event['positions'])
  times     = np.array(event['times'])
  times     = np.where(np.isnan(times), 0.0, times)
  beta      = float(event.get('beta', 0.0))

  layer_idx = (vids // 1000000).astype(np.int64)
  is_tof    = layer_idx < 200

  tof_e   = energies[is_tof]
  tof_pos = positions[is_tof]
  tof_t   = times[is_tof]
  tof_li  = layer_idx[is_tof] % 100

  is_outer = tof_li < 6
  is_inner = ~is_outer

  outer_e = float(tof_e[is_outer].sum()) if is_outer.any() else 0.0
  inner_e = float(tof_e[is_inner].sum()) if is_inner.any() else 0.0
  outer_n = float(is_outer.sum())
  inner_n = float(is_inner.sum())

  valid_t = tof_t[tof_t > 0]
  tof_range = float(valid_t.max() - valid_t.min()) if len(valid_t) > 1 else 0.0

  if len(tof_pos) > 0:
      valid_mask = tof_t > 0
      if valid_mask.any():
          first = int(np.argmin(np.where(valid_mask, tof_t, np.inf)))
      else:
          first = 0
      entry_x = float(tof_pos[first, 0])
      entry_y = float(tof_pos[first, 1])
      entry_z = float(tof_pos[first, 2])
  else:
      entry_x = entry_y = entry_z = 0.0

  si_e = float(energies[~is_tof].sum())

  return np.array([
      outer_e   / 100.0,
      inner_e   / 100.0,
      outer_n   / 10.0,
      inner_n   / 10.0,
      tof_range / 50.0,
      entry_x   / 1000.0,
      entry_y   / 1000.0,
      entry_z   / 1000.0,
      float(is_tof.sum()) / 20.0,
      si_e      / 500.0,
      beta,
  ], dtype=np.float32)


if __name__ == '__main__':
    np.random.seed(0)
    n_sili, n_tof = 50, 20

    sili_vids = ((200 + np.random.randint(0, 10, n_sili)) * 1_000_000).tolist()
    tof_vids = (np.random.randint(0, 6, n_tof) * 1_000_000).tolist()
    all_vids = sili_vids + tof_vids
    n_total = len(all_vids)

    mock_event = {
        'volume_id': all_vids,
        'energy': np.random.uniform(0.1, 5.0, n_total).tolist(),
        'positions': np.random.uniform(-600, 600, (n_total, 3)).tolist(),
        'times': np.concatenate([np.full(n_sili, np.nan),
                                 np.random.uniform(1, 100, n_tof)]).tolist(),
        'beta': 0.35,
        'label': -1000010020,
    }

    voxel = build_sili_voxel(mock_event)
    tof = build_tof_features(mock_event)

    print(f"voxel shape: {voxel.shape}")  # 期望: (10, 20, 20)
    print(f"voxel dtype: {voxel.dtype}")  # 期望: float32
    print(f"voxel max:   {voxel.max():.4f}")  # 期望: > 0
    print(f"nonzero bins: {(voxel > 0).sum()}")
    print(f"tof shape:   {tof.shape}")  # 期望: (11,)
    print(f"tof values:  {np.round(tof, 4)}")
    print(f"has nan/inf: {np.isnan(tof).any() or np.isinf(tof).any()}")  # 期望: False


"""
voxel shape: (10, 20, 20)
voxel dtype: float32
voxel max:   5.3258
nonzero bins: 49
tof shape:   (11,)
tof values:  [ 0.5386  0.      2.      0.      1.8126 -0.1064 -0.1689 -0.1206  1.
  0.287   0.35  ]
has nan/inf: False
"""