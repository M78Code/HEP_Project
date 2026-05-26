import numpy as np

GRID_Z, GRID_X, GRID_Y = 10, 12, 12   # layers × x × y（中上と同じ軸順）
X_MIN, X_MAX = -700.0, 700.0
Y_MIN, Y_MAX = -700.0, 700.0

def build_sili_voxel(event):
  """Si(Li)ヒット → 3Dボクセルグリッド (10,12,12) = layers×x×y"""
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
  """TOF関連11次元特徴量（中上 Table 6.1 準拠）
  [outer_e, inner_e, outer_n, inner_n, time_of_flight,
   outer_entry_x, outer_entry_y, outer_entry_z,
   inner_entry_x, inner_entry_y, inner_entry_z]
  """
  vids      = np.array(event['volume_id'])
  energies  = np.array(event['energy'])
  positions = np.array(event['positions'])
  times     = np.array(event['times'])
  times     = np.where(np.isnan(times), 0.0, times)

  layer_idx = (vids // 1000000).astype(np.int64)
  is_tof    = layer_idx < 200

  tof_e   = energies[is_tof]
  tof_pos = positions[is_tof]
  tof_t   = times[is_tof]
  tof_li  = layer_idx[is_tof] % 100

  is_outer = tof_li < 6      # layer 10X: 0,2,3,4,5
  is_inner = ~is_outer        # layer 11X: 10,11,12,13,14,15,16

  # 1-2: outer/inner energy loss
  outer_e = float(tof_e[is_outer].sum()) if is_outer.any() else 0.0
  inner_e = float(tof_e[is_inner].sum()) if is_inner.any() else 0.0

  # 3-4: outer/inner paddle通過数（unique volume_id数）
  tof_vids = vids[is_tof]
  outer_n = float(len(np.unique(tof_vids[is_outer]))) if is_outer.any() else 0.0
  inner_n = float(len(np.unique(tof_vids[is_inner]))) if is_inner.any() else 0.0

  # 5: Time of Flight（outer最早hit → inner最早hit の時間差）
  outer_t = tof_t[is_outer]
  inner_t = tof_t[is_inner]
  outer_valid = outer_t[outer_t > 0]
  inner_valid = inner_t[inner_t > 0]
  if len(outer_valid) > 0 and len(inner_valid) > 0:
      tof_value = float(inner_valid.min() - outer_valid.min())
  else:
      tof_value = 0.0

  # 6-8: outer TOF 入射座標（最早hit位置）
  if is_outer.any() and len(outer_valid) > 0:
      outer_pos = tof_pos[is_outer]
      outer_first = int(np.argmin(np.where(outer_t > 0, outer_t, np.inf)))
      outer_entry = outer_pos[outer_first]
  else:
      outer_entry = np.zeros(3)

  # 9-11: inner TOF 入射座標（最早hit位置）
  if is_inner.any() and len(inner_valid) > 0:
      inner_pos = tof_pos[is_inner]
      inner_first = int(np.argmin(np.where(inner_t > 0, inner_t, np.inf)))
      inner_entry = inner_pos[inner_first]
  else:
      inner_entry = np.zeros(3)

  return np.array([
      outer_e   / 100.0,
      inner_e   / 100.0,
      outer_n   / 10.0,
      inner_n   / 10.0,
      tof_value / 50.0,
      outer_entry[0] / 1000.0,
      outer_entry[1] / 1000.0,
      outer_entry[2] / 1000.0,
      inner_entry[0] / 1000.0,
      inner_entry[1] / 1000.0,
      inner_entry[2] / 1000.0,
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
voxel shape: (10, 12, 12)
voxel dtype: float32
voxel max:   5.3258
nonzero bins: 49
tof shape:   (11,)
tof values:  [ 0.5386  0.      0.6     0.      0.     -0.1064 -0.1689 -0.1206  0.
  0.      0.    ]
has nan/inf: False
"""