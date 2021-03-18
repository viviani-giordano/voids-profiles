# Color Class

class col:
    """
    Class that defines some of the terminal colors useful while printing:
    Possible colors:
        -HEADER
        -STEP
        -WARNING
        -FAIL
        -IMPORTANT_FAIL
        -UNDERLINE
        -END (to restore default printing setting)
    """
    HEADER = '\x1b[32m'
    HEADER2 = '\x1b[1;33m'
    NOTICE = '\x1b[33m'
    STEP = '\x1b[36m'
    WARNING = '\x1b[43m'
    FAIL = '\x1b[0;31;47m'
    IMPORTANT_FAIL = '\x1b[7;31;47m'
    UNDERLINE = '\x1b[4m'
    END = '\x1b[0m'


    def colors(num):
        """
        Function used to compute a certain number (num) of colors
        shading from red to green through blue.

        Return: array of colors (size=num)
                
        """
        color = []
        for i in range(num):
            ratio = i/num
            r = max(1-(2*ratio),0)
            g = max(2*ratio-1,0)
            b = min(2*ratio, -2*ratio+2)
            color.append((r,g,b))

        #color = []
        #for i in range(num):
        #    ratio = i/num
        #    r = max(1-(ratio),0)
        #    g = max(ratio-1,0)
        #    b = min(2*ratio, -2*ratio+2)
        #    color.append((r,g,b))
        return color    

