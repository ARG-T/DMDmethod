# 実行ファイル
import DMD
import crystal_structure

if __name__ == "__main__":
    lat_parameter, lat_x, lat_y, lat_z, temperature, mass = map(int, input().split())
    structure = input()
    atom_position = crystal_structure.det_structure(structure)
    if atom_position is False:
        print("Error !")
        exit()
    DMD_analysis = DMD.DMDmethod(lat_parameter, lat_x, lat_y, lat_z, atom_position, temperature, mass)
    DMD_analysis.main_roop()
