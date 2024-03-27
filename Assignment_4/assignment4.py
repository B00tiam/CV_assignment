from train_validate import train_validate
from test import test

if __name__ == "__main__":
    num_epoch = 15

    print("Pls input the model you want to train and test(press): baseline(0), var1(1), var2(2), var3(3), var4(4)")
    choice1_str = input("Your choice:")
    choice1_num = int(choice1_str)

    # Run the code:
    print("Pls choose the choice task you want to run(press): basic(0), k-fold(1), data augmentation(2), t-SNE(3), lr decrease(4)")
    choice2_str = input("Your choice:")
    choice2_num = int(choice2_str)
    if choice2_num == 0:
        train_validate(num_epoch, choice1_num, is_lr=0, is_improve=0)
        test(choice1_num, is_improve=0)
    if choice2_num == 2:
        train_validate(num_epoch, choice1_num, is_lr=0, is_improve=1)
        test(choice1_num, is_improve=1)
    if choice2_num == 4:
        train_validate(num_epoch, choice1_num, is_lr=1, is_improve=0)
        test(choice1_num, is_improve=0)
    else:
        exit(0)
