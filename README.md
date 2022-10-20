# AIM22-ReverseISP-MiAlgo

PyTorch implementation of AIM 2022 reverse ISP challenge by MiAlgo team.

## Dependencies

* Python 3
* PyTorch >= 1.8
* numpy
* cv2
* natsort
* glob

## Test
 ```bash
    # Test our trained model
    
    # For s7(track1)
    python3 test.py --input_dir "path/to/input/dir" --output_dir "path/to/output/dir" --checkpoint 'ckpts/s7.pth' --blevl 0 --wlevl 1023
    #e.g.
    python3 test.py --input_dir "data/s7_test_rgb_demo" --output_dir "data/s7_test_result" --checkpoint 'ckpts/s7.pth' --blevl 0 --wlevl 1023
    
     # For p20(track2)
    python3 test.py --input_dir "path/to/input/dir" --output_dir "path/to/output/dir" --checkpoint 'ckpts/p20.pth' --blevl 60 --wlevl 1020
    #e.g.
    python3 test.py --input_dir "data/p20_test_rgb_demo" --output_dir "data/p20_test_result" --checkpoint 'ckpts/p20.pth' --blevl 60 --wlevl 1020
    
 ```

