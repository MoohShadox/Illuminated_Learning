#!/usr/bin/python -w

import sys
import matplotlib.pyplot as plt

goal=[60, 60]

def plot_points(points, bg="maze_hard.pbm", title=None):
    x,y = zip(*points)
    fig1, ax1 = plt.subplots()
    ax1.set_xlim(0,600)
    ax1.set_ylim(600,0) # Decreasing
    ax1.set_aspect('equal')
    if(bg):
        img = plt.imread(bg)
        ax1.imshow(img, extent=[0, 600, 600, 0])
    if(title):
        ax1.set_title(title)
    ax1.scatter(x, y, s=2)
    ax1.scatter([goal[0]],[goal[1]], c="red") 
    plt.show()
    
def plot_points_lists(lpoints, bg="maze_hard.pbm", title=None):
    fig1, ax1 = plt.subplots()
    ax1.set_xlim(0,600)
    ax1.set_ylim(600,0) # Decreasing
    ax1.set_aspect('equal')
    if(bg):
        img = plt.imread(bg)
        ax1.imshow(img, extent=[0, 600, 600, 0])
    if(title):
        ax1.set_title(title)
    for points in lpoints:
        x,y = zip(*points)
        ax1.scatter(x, y, s=2)
    plt.show()


def plot_traj_file(filename, bg="maze_hard.pbm", title=None):
    try:
        with open(filename) as f:
            points=[]
            for l in f.readlines():
                try:
                    l=l.strip()
                    l=l.strip(" ")
                    pos=list(map(float, l.split(" ")))
                    points.append(pos[0:2]) # we discard the angle
                except ValueError:
                    print("Invalid line: \""+l+"\"")

            f.close()
            plot_points(points, bg, title)
    except IOError:
        print("Could not read file: "+f)

if __name__ == '__main__':
    for arg in sys.argv[1:]:
        plot_traj_file(arg, title=arg)
