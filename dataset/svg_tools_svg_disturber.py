#!/usr/bin/env python
""" svg_disturber.py takes a single svg file in input and disturb paths and polylines in it to output a given number of disturbed svg """

import sys
import re
import numpy as np
from PIL import Image
# import svgwrite
import random 
import argparse
import uuid
from math import pi, cos, sin
from xml.dom import minidom
import json

F = random.uniform(0.0, 1.0)

def getPolylineData(p):
    poly = {}

    points = p.getAttribute('points').split()
    points = map(lambda p: tuple(map(float, p.split(','))), points)
    point_list = [[x[0],x[1]] for x in points]
    poly['points'] = point_list

    center = [0,0]
    for x in point_list: center=np.sum([center,x], axis=0)
    poly['center'] = np.array(center)/len(point_list)

    return poly

def m_rot(a):
    return np.matrix([[cos(a), -sin(a), 0], 
                      [sin(a), cos(a),  0],
                      [0,      0,       1]])

def m_trans(x, y):
    return np.matrix([[1, 0, x], 
                      [0, 1, y],
                      [0, 0, 1]])

def m_scale(sx, sy):
    return np.matrix([[sx, 0, 0], 
                      [0, sy, 0],
                      [0, 0,  1]])

def random_translate(max_dist):
    theta = random.uniform(-pi, pi)
    norm = random.uniform(0, max_dist) * F
    tx, ty = norm*cos(theta), norm*sin(theta)
    
    return m_trans( tx, ty )

def random_scale(minimum, maximum, par):
    minimum = 1 + (minimum - 1)* F        
    maximum = 1 + (maximum - 1)* F   

    if par:
        r1 = random.uniform(minimum, maximum) 
        r2 = r1
    else:
        r1 = random.uniform(minimum, maximum)
        r2 = random.uniform(minimum, maximum)
    
    return m_scale(r1, r2)

def getRandomTransform(center, params):
    MAX_ANGLE = params['MAX_ANGLE']
    MIN_SCALE = params['MIN_SCALE']
    MAX_SCALE = params['MAX_SCALE']
    MAX_TRANSLATE = params['MAX_TRANSLATE']
    PRESERVE_RATIO = params['PRESERVE_RATIO']

    T_ori = m_trans( -center[0], -center[1] )
    
    a = random.uniform(-MAX_ANGLE, MAX_ANGLE) * F

    R = m_rot( a )
    S = random_scale( MIN_SCALE, MAX_SCALE, PRESERVE_RATIO )
    T_inv = m_trans( center[0], center[1] )
    T = random_translate( MAX_TRANSLATE )

    M = T * T_inv * S * R * T_ori

    return M

def disturbPoly(poly, M_local=np.identity(3), noise=0):
    points = []
    for v in poly['points']:
        P = np.matrix([v[0], v[1], 1]).transpose()
        T_noise = random_translate(noise)
        p_rot = T_noise * M_local * P
        points.append([p_rot.item(0), p_rot.item(1)])
    return points

def coherentDisturb(a, b, p):
    u = np.array(b)-np.array(a)
    v = np.array([-u[1], u[0]])
    if np.linalg.norm(v) != 0:
        v = v/np.linalg.norm(v)
    return (b + random.uniform(-p, p) * F * v).tolist()

def addOverstroke(data, p, under=False):
    b, c = np.array(data[0]), np.array(data[1])
    y, x = np.array(data[len(data) - 1]), np.array(data[len(data) - 2])

    u = b - c; 
    nu = np.linalg.norm(u)
    if nu > 0:
        u = u/nu
    else:
        u = np.array([0,0])    
    r = random.uniform(max(-p, -nu) if under else 0, p) * F
    # print b, "+", r, u
    a = b + r * u

    v = y - x
    nv = np.linalg.norm(v)
    if nv > 0:
        v = v/nv
    else:
        v = np.array([0,0]) 
    s = random.uniform(max(-p, -nv) if under else 0, p) * F
    # print y, "+", s, v
    z = y + s * v

    if r > 0:
        data.insert(0, [a[0], a[1]])
    else:
        data[0] = [a[0], a[1]]

    if s > 0:
        data.append([z[0], z[1]])
    else:
        data[len(data)-1] = [z[0], z[1]]

def addCoherentNoise(data, p):
    new_data = []

    o = coherentDisturb(data[1], data[0], p)
    new_data.append(o)

    prev = data[0]
    for pt in data[1:]:
        new_data.append(coherentDisturb(prev, pt, p))
        prev = pt

    return new_data

def flatten(l):
    return [item for sublist in l for item in sublist]

def parseAndDisturb(in_node, out_parent, out_svg, params):
    if in_node.nodeType != 1: return

    # print in_node.nodeName, in_node.nodeType, in_node.nodeValue
    if in_node.nodeName == 'polyline':
        for i in range(random.randint(params['MIN_STROKES'], params['MAX_STROKES'])):
            out_node = out_svg.createElement(in_node.nodeName)
            data = getPolylineData(in_node)
            M_local = getRandomTransform(data['center'], params)

            disturbed_data = disturbPoly(data, M_local, 0 if params['COHERENT'] else params['PER_POINT_NOISE'])

            if params['OVERSTROKE'] > 0:
                addOverstroke(disturbed_data, params['OVERSTROKE'], params['UNDERSTROKE'])

            if params['COHERENT']:
                disturbed_data = addCoherentNoise(disturbed_data, params['PER_POINT_NOISE'])

            for a in in_node.attributes.keys():
                if a == 'points':
                    out_node.setAttribute(a, ''.join("%0.3f,%0.3f"%(x[0],x[1])+' ' for x in disturbed_data))
                elif a == 'stroke-width' and params['PEN'] > 0:
                    out_node.setAttribute(a, str(params['PEN']))
                else:
                    out_node.setAttribute(a, in_node.getAttribute(a))  
            out_node.setAttribute("stroke-linejoin", "round")
            out_parent.appendChild(out_node)
    elif in_node.nodeName == 'path':
        out_node = out_svg.createElement(in_node.nodeName)
        for a in in_node.attributes.keys():
            if a == 'stroke-width' and params['PEN'] > 0: 
                out_node.setAttribute(a, str(params['PEN']))            
            else:
                out_node.setAttribute(a, in_node.getAttribute(a))
        out_parent.appendChild(out_node)        
    else:
        out_node = out_svg.createElement(in_node.nodeName)
        for a in in_node.attributes.keys():
            out_node.setAttribute(a, in_node.getAttribute(a))
        out_parent.appendChild(out_node)

    if in_node.nodeName == 'svg' and params['BG']:
        rect = out_svg.createElement('rect')
        rect.setAttribute('width', in_node.getAttribute("width"))
        rect.setAttribute('height', in_node.getAttribute("height"))
        rect.setAttribute('style', 'fill:rgb(255,255,255);')  
        out_node.appendChild(rect) 

    for child in in_node.childNodes:
        parseAndDisturb(child, out_node, out_svg, params)


def main():
    global F
    parser = argparse.ArgumentParser()
    parser.add_argument("svg_in", help="svg input file")
    parser.add_argument("out_file", help="output svg file")    
    parser.add_argument("-a", help="flag to unforce aspect ratio locally (for scale action)", action="store_true")    
    parser.add_argument("-c", help="flag for coherent noise", action="store_true")    
    parser.add_argument("-n", help="global noise applied to every point (translation upper bound)", type=float, default=0)    
    parser.add_argument("-r", help="local rotation angle upper bound (degrees)", type=float, default=0)
    parser.add_argument("-sl", help="local scale factor lower bound", type=float, default=1)
    parser.add_argument("-su", help="local scale factor upper bound", type=float, default=1)
    parser.add_argument("-t", help="local translate distance upper bound", type=float, default=0)
    parser.add_argument("-min", help="minimum nb of strokes for oversketching", type=int, default=1)
    parser.add_argument("-max", help="maximum nb of strokes for oversketching", type=int, default=1)
    parser.add_argument("-os", help="overstroke max norm", type=float, default=0)
    parser.add_argument("-u", help="flag to add understroke in overstroke", action="store_true")
    parser.add_argument("-pen", help="override pen size", type=float, default=-1)
    parser.add_argument("-penv", help="pen size variance", type=float, default=0)
    parser.add_argument("-bg", help="flag to add white background", action="store_true")  
    parser.add_argument("-f", help="flag to use global noise parameter [0-1]", action="store_true")  

    a = parser.parse_args()
    a.r = a.r*pi/180

    params = { }
    params['MAX_ANGLE'] = a.r               # angle in radian
    params['MIN_SCALE'] = a.sl              # scale ratio
    params['MAX_SCALE'] = a.su              # scale ratio
    params['MAX_TRANSLATE'] = a.t           # translation distance in svg units
    params['PRESERVE_RATIO'] = not a.a      # bool  
    params['PER_POINT_NOISE'] = a.n         # absolute translation distance in svg units   
    params['MIN_STROKES'] = a.min           # minimum number of strokes
    params['MAX_STROKES'] = a.max           # maximum //
    params['OVERSTROKE'] = a.os             # bool
    params['COHERENT'] = a.c                # bool
    params['UNDERSTROKE'] = a.u             # bool
    params['PEN'] = a.pen                   # pen size in svg units
    params['BG'] = a.bg                     # bool

    if params['PEN'] > 0 and a.penv > 0:
        a.penv = a.penv * F
        params['PEN'] = random.uniform(params['PEN'] - a.penv/2.0, params['PEN'] + a.penv/2.0)
        
    params['PENV'] = a.penv                 # pen size in svg units

    if not a.f:
        F = 1.0
    params['F'] = F

    svg_in = minidom.parse(a.svg_in)
    svg_out = minidom.Document()

    parseAndDisturb(svg_in.documentElement, svg_out, svg_out, params)

    out_file = open(a.out_file, 'w')
    out_file.write(svg_out.toprettyxml())
    out_file.close()

    # json.dump(params, open(a.out_file+".json", 'w'), indent=4, sort_keys=True)
    

if __name__ == '__main__':
    main()


