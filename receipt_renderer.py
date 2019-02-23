import cv2
from matplotlib.pyplot import imshow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from synthgen import RendererV3
import random
import re
import text_utils as tu
from text_utils import crop_safe
import pygame
import argparse
import h5py
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--out-file', type=str)

def ims(img):
    imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def bb_xywh2coords(bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n,_ = bbs.shape
        coords = np.zeros((2,4,n))
        for i in xrange(n):
            coords[:,:,i] = bbs[i,:2][:,None]
            coords[0,1,i] += bbs[i,2]
            coords[:,2,i] += bbs[i,2:4]
            coords[1,3,i] += bbs[i,3]
        return coords

class ReceiptGenerator:
    def __init__(self):
        with open('data/product_codes/products_with_prices.txt', 'r') as f:
            self.prod_prices = f.read().strip().split('\n')
        with open('data/product_codes/categories.txt', 'r') as f:
            self.categories = f.read().strip().split('\n')
        with open('data/product_codes/products_with_prices_member.txt', 'r') as f:
            self.prod_prices_member = f.read().strip().split('\n')

    def get_receipt(self, num_lines):
        
        member = random.random() < .8

        num_categories = min(num_lines // 4, len(self.categories))
        num_product_prices = num_lines - (2 * num_categories)

        categories = random.sample(self.categories, num_categories)

        if member:
            random_start = np.random.randint(0, len(self.prod_prices_member) - num_lines)
            all_lines = self.prod_prices_member[random_start:random_start+num_lines]
        else:

            random_start = np.random.randint(0, len(self.prod_prices) - num_lines)
            all_lines = self.prod_prices[random_start:random_start+num_lines]
        if num_categories == 1:
            return (member, zip(categories, all_lines))
        else:
            out_prices = []
            random_segments = sorted(random.sample(range(1, num_product_prices), num_categories - 1))

            start = 0
            for i in random_segments:
                out_prices.append(all_lines[start:i])
                start = i
            out_prices.append(all_lines[start:])
            return (member, zip(categories, out_prices))

    def render_multiline(self, font, num_lines, fsize, char_width=6, full_width=300):
        """
        renders multiline TEXT on the pygame surface SURF with the
        font style FONT.
        A new line in text is denoted by \n, no other characters are 
        escaped. Other forms of white-spaces should be converted to space.
        returns the updated surface, words and the character bounding boxes.
        """
        
        is_member, receipt_fields = self.get_receipt(num_lines) 

        W, H = fsize
        
        # font parameters:
        line_spacing = font.get_sized_height() + 1

        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        bbs = []
        space = font.get_rect('O')
       
        y = 0
        
        words = []
        
        for section, prices in receipt_fields:
            # place section
            x = full_width / 2 - (len(section) // 2) * char_width
            y += line_spacing
            
            words.append(section)
            
            font.strong = True
            y, bbs, surf = place_lines([section], font, surf, bbs, x, y, char_width, line_spacing, True)
            font.strong = False
            x = char_width * 10
            y += line_spacing
        
            y, bbs, surf = place_lines(prices, font, surf, bbs, x, y, char_width, line_spacing, False)
            
            words.extend(prices)

        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)
        # crop the surface to fit the text:
        bbs = np.array(bbs)
        # _, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
        surf_arr = pygame.surfarray.pixels_alpha(surf)
        surf_arr = surf_arr.swapaxes(0,1)
        return surf_arr, words, bbs, '\n'.join(words)
    
def place_lines(lines, font, surf, bbs, default_x, y, char_width, line_spacing, is_title):
    for l in lines:
        if not is_title:
            font.strong = False
            for cc in l:
                if cc.islower():
                    font.strong = True
                    break
        x = default_x
        y += line_spacing # line-feed
        for ch in l: # render each character
            if ch.isspace(): # just shift
                x += char_width
            else:
                ch_bounds = font.render_to(surf, (x,y), ch)
                ch_bounds.x = x + ch_bounds.x
                ch_bounds.y = y - ch_bounds.y

                x += char_width
                bbs.append(np.array(ch_bounds))
    return y, bbs, surf

def get_mask_transform(mask, text_mask):
    aa = mask[['x','y']].values.astype(np.int32)
    zeros = np.zeros(text_mask.shape)
    cv2.fillPoly(zeros, np.array([aa]), 1)
    text_mask = text_mask * zeros
    return text_mask

def add_res(imgname,res, i):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    name = imgname.split('.')[0] + '_' + str(i).zfill(4) + '.jpeg'
    img = res['img']
    reshaped = np.round(res['wordBB'].swapaxes(2, 0).reshape(-1, 8), 3)
    cv2.imwrite('/data0/ocr/rot_boxes_v2/images/'+name, img)
    np.savetxt('/data0/ocr/rot_boxes_v2/boxes/'+name.replace('.jpeg', '.txt'), reshaped, delimiter=",", fmt='%-10.5f')

if __name__ == '__main__':
    args = parser.parse_args()

    num_instances = 500
    planes = pd.read_csv('receipts_02_22/plane.txt', header=None)
    planes.columns = ['path','x','y','surf_num']
    planes = planes.groupby('path').filter(lambda x: len(x) % 4 == 0)

    masks = pd.read_csv('receipts_02_22/mask.txt', header=None)
    masks.columns = ['path','x','y','surf_num']
    masks = masks[['path','x','y']]

    masks = masks.groupby('path')

    DEFAULT_WIDTH = 300

    text_renderer = tu.RenderFont('data/')

    receipt_gen = ReceiptGenerator()
    RV3 = RendererV3('data/')
    for name, grp in tqdm(planes.groupby('path')):
        orig_img = cv2.imread(name)
        imname = os.path.basename(name) 
        if name in masks.groups.keys():
            mask = masks.get_group(name)
            continue
        else:
            mask = None
            
        surf_stuff = []
        for surf_num, grp2 in grp.groupby('surf_num'):
            pts_src = grp2[['x','y']].values
            rect = cv2.minAreaRect(pts_src)

            rot_h, rot_w = rect[1]
            
            w = min(rot_h, rot_w)
            h = max(rot_h, rot_w)
            
            aspect_ratio = w / h
            
            width = DEFAULT_WIDTH
            height = DEFAULT_WIDTH / aspect_ratio
            
            width = int(width)
            height = int(height)
            
            pts_dst = np.array(((0,0),(width-1,0),(width-1,height-1),(0,height-1)))
            
            H, _ = cv2.findHomography(pts_dst, pts_src, method=0)
            Hinv, _ = cv2.findHomography(pts_src, pts_dst, method=0)
            
            surf_stuff.append({'width': width, 'height': height, 'H': H, 'Hinv': Hinv})
        
        for i in range(num_instances):
            img = orig_img.copy()

            font = text_renderer.font_state.sample(True)
            font['font'] = 'data/fonts/receipt_fonts/receipt9.ttf'
            font['size'] = 11
            font['kerning'] = True
            itext = []
            ibb = []
            idict = {'img':[], 'charBB':None, 'wordBB':None, 'txt':None}

            font_renderer = text_renderer.font_state.init_font(font)

            for surf in surf_stuff:
                width = surf['width']
                height = surf['height']
                H = surf['H']
                Hinv = surf['Hinv']
                
                num_lines = (height // (font_renderer.get_sized_height() + 1)) / 2
                if num_lines < 3:
                    continue

                text_mask, words, bb, text = receipt_gen.render_multiline(font_renderer, num_lines, (width, height))
                text = re.sub(' +', ' ', ' '.join(words))
                
                bb = bb_xywh2coords(bb)
                bb = RV3.homographyBB(bb,H)
                
                min_h = RV3.get_min_h(bb,text)
                text_mask = RV3.feather(text_mask, min_h)        
                text_mask = cv2.warpPerspective(text_mask, H, (img.shape[1],img.shape[0]))
                
                if mask is not None:
                    text_mask = get_mask_transform(mask, text_mask)

                img = RV3.colorizer.color(img,[text_mask],np.array([min_h]))
                cv2.imwrite('checking/'+name.split('/')[1], img)
                
                itext.append(text)
                ibb.append(bb)
            
            idict['img'] = img
            idict['txt'] = itext
            idict['charBB'] = np.concatenate(ibb, axis=2)
            idict['wordBB'] = RV3.char2wordBB(idict['charBB'].copy(), ' '.join(itext))
            add_res(imname, idict, i)
