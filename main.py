# 実行ファイル
import DMD

if __name__ == "__main__":
    lat_parameter, lat_x, lat_y, lat_z, temperature, mass = map(int,input().split())
    DMD_analysis = DMD.DMDmethod(lat_parameter, lat_x, lat_y, lat_z, temperature, mass)
    DMD_analysis.main_roop()