import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab()
print "Matlab engine started..."
ratings = np.array(eng.STB_pred(1, 1))
print "ratings: " 
print ratings

eng.quit()
