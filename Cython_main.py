from c_free_energy import FreeEnergy


def culc_lat(temperature):
    print(temperature)
    lat_p = 3.65
    temperature = temperature
    sigma = pow(10, -5)
    learning_rate = 0.1
    while True:
        FE1 = FreeEnergy(lat_p+sigma, temperature)
        FE2 = FreeEnergy(lat_p-sigma, temperature)
        free1 = FE1.main_loop()
        free2 = FE2.main_loop()
        slope = (free1-free2)/(2*sigma)
        delta = slope*learning_rate
        if abs(delta) < pow(10, -7):
            return lat_p, temperature
            break
        if abs(delta) < pow(10, -6):
            learning_rate = 0.05
        lat_p -= delta


if __name__ == "__main__":
    result = culc_lat(200)
    print(result)
