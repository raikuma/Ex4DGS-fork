import sys
from plyfile import PlyData
import os

model_path = sys.argv[1]
iteration = sys.argv[2]

ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
ply_data = PlyData.read(ply_path)
static = len(ply_data['vertex'])

ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "dynamic_point_cloud.ply")
ply_data = PlyData.read(ply_path)
dyn = len(ply_data['vertex'])

total = static + dyn

with open(os.path.join(model_path, "number.txt"), 'w') as f:
    f.write(f"Anchor: {0}\nTotal: {total}\nActive: {total}\nRatio: {0}\nStatic: {total - dyn}\nDynamic: {dyn}")
    print(f"Anchor: {0}, Total: {total}, Active: {total}, Ratio: {0}, Static: {total - dyn}, Dynamic: {dyn}")