import numpy as np

FRAME_RATE = 3  # frames per second
NUM_SAMPLES_PER_GEN = 5  

dims = np.array([])
    
for i in range(1, 200):
    gen = np.load(f"ga_population_gen_{i}.npy")
    for j in range(NUM_SAMPLES_PER_GEN):
        dims = np.append(dims, gen[j])
        frames = fill_in_dims(dims1, dims2)  # (2*FRAME_RATE, dim)
        np.save(f"animation/frames/gen_{i}_ind_{j}.npy", frames)




def fill_in_dims(dims1:np.ndarray, dims2:np.ndarray)-> np.ndarray:


    trasform = np.linspace(dims1, dims2, FRAME_RATE, axis=0)  # (FRAME_RATE, dim)
    still = np.tile(dims2, (FRAME_RATE, 1))
    fill = np.vstack((trasform, still))  # (2*FRAME_RATE, dim)
    return fill 

if __name__ == "__main__":
    dims1 = np.array([1.0, 2.0, 3.0])
    dims2 = np.array([4.0, 5.0, 6.0])
    trasform = fill_in_dims(dims1, dims2)

    print(trasform)
    print(trasform.shape)  # should be (FRAME_RATE, dim)

