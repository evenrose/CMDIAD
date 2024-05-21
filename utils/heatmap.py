import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# main_xyz_path = 'D:/segmentation/main_xyz/cable_gland/test/thread/rgb/013.pt'
# main_xyz_path = 'D:/segmentation/main_xyz/foam/test/contamination/rgb/013.pt'
# main_xyz_path = 'D:/segmentation/main_xyz/cookie/test/crack/rgb/003.pt'

# main_xyz_path = 'D:/segmentation/main_xyz/cookie/test/combined/rgb/002.pt'
# main_xyz_path = 'D:/segmentation/main_xyz/cookie/test/hole/rgb/005.pt'

# main_xyz_path = 'D:/segmentation/main_xyz/cable_gland/test/hole/rgb/021.pt'
# main_xyz_path = 'D:/segmentation/main_xyz/cable_gland/test/thread/rgb/006.pt'
# main_xyz_path = 'D:/segmentation/main_xyz/cable_gland/test/hole/rgb/008.pt'
main_xyz_path = 'D:/segmentation/main_xyz/cable_gland/test/bent/rgb/010.pt'

main_xyz = torch.load(main_xyz_path)[0]

main_rgb_path = main_xyz_path.replace('main_xyz', 'main_rgb')
main_rgb = torch.load(main_rgb_path)[0]

rgb_path = main_xyz_path.replace('main_xyz', 'rgb')
rgb = torch.load(rgb_path)[0]

xyz_path = main_xyz_path.replace('main_xyz', 'xyz')
xyz = torch.load(xyz_path)[0]
data = [main_xyz, main_rgb, xyz, rgb]

fig, axs = plt.subplots(2,2, figsize=(10, 10), dpi=300)

sns.heatmap(data=main_xyz, ax=axs[1, 0], xticklabels=[], yticklabels=[])
sns.heatmap(data=main_rgb, ax=axs[1, 1], xticklabels=[], yticklabels=[])
sns.heatmap(data=xyz, ax=axs[0, 0], xticklabels=[], yticklabels=[])
sns.heatmap(data=rgb, ax=axs[0, 1], xticklabels=[], yticklabels=[])
# sns.heatmap(data=main_xyz, ax=axs[1, 0], cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[], vmax=0.000062)
# sns.heatmap(data=main_rgb, ax=axs[1, 1], cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# sns.heatmap(data=xyz, ax=axs[0, 0], cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[], vmax=0.000062)
# sns.heatmap(data=rgb, ax=axs[0, 1], cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=main_xyz, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('cable_gland_thread_013_main_xyz.png', bbox_inches='tight', transparent=True)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=main_rgb, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('cable_gland_thread_013_main_rgb.png', bbox_inches='tight', transparent=True)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=xyz, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('cable_gland_thread_013_xyz.png', bbox_inches='tight', transparent=True)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=rgb, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('foam_contamination_013_rgb.png', bbox_inches='tight', transparent=True)

# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=main_xyz, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('foam_contamination_013_main_xyz.png', bbox_inches='tight', transparent=True)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=main_rgb, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('foam_contamination_013_main_rgb.png', bbox_inches='tight', transparent=True)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=xyz, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('foam_contamination_013_xyz.png', bbox_inches='tight', transparent=True)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=rgb, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('foam_contamination_013_rgb.png', bbox_inches='tight', transparent=True)

# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=main_xyz, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('cookie_crack_003_main_xyz.png', bbox_inches='tight', transparent=True)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=main_rgb, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('cookie_crack_003_main_rgb.png', bbox_inches='tight', transparent=True)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=xyz, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('cookie_crack_003_xyz.png', bbox_inches='tight', transparent=True)
# fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
# sns.heatmap(data=rgb, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
# plt.savefig('cookie_crack_003_rgb.png', bbox_inches='tight', transparent=True)

# filename = 'cookie_combined_002'
# filename = 'cookie_hole_005'
# filename = 'cable_gland_hole_021'
# filename = 'cable_gland_thread_006'
# filename = 'cable_gland_hole_008'
filename = 'cable_gland_bent_010'
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
sns.heatmap(data=main_xyz, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
plt.savefig(filename+'_main_xyz.png', bbox_inches='tight', transparent=True)
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
sns.heatmap(data=main_rgb, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
plt.savefig(filename+'_main_rgb.png', bbox_inches='tight', transparent=True)
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
sns.heatmap(data=xyz, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
plt.savefig(filename+'_xyz.png', bbox_inches='tight', transparent=True)
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
sns.heatmap(data=rgb, ax=ax, cmap="YlGnBu_r", cbar=False, xticklabels=[], yticklabels=[])
plt.savefig(filename+'_rgb.png', bbox_inches='tight', transparent=True)

