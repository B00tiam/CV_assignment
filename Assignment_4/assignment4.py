from train_validate import train_validate
from test import test

if __name__ == "__main__":
    num_epoch = 10

    print("Pls input the model you want to train and test(press): baseline(0), var1(1), var2(2), var3(3), var4(4)")
    choice_str = input("Your choice:")
    choice_num = int(choice_str)

    # Run the code:
    train_validate(num_epoch, choice_num)
    test(choice_num)