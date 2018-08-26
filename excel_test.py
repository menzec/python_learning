# -*- coding: utf-8 -*-
# @Date    : 2018-07-27 17:26:39
# @Author  : ${menzec} (${menzc@outlook.com})
# @Link    : http://example.org
# @Version : $Id$

import sys
import shutil
import os
import xlrd
import xlwt


def Regular_Dir(root):
    if os.path.exists(root):
        if root[-1] not in ['\\', '/']:
            root += ('/')
            return root
    else:
        return False


def GetFileList(root, file_extensions, absolute_dir=True):
    '''返回的文件夹下的文件，Dir = Ture表示返回绝对路径'''
    if not Regular_Dir(root):
        print(root, " dir is not exists!")
        sys.exit()
    filelist = []
    for files in os.listdir(root):
        if os.path.isdir(files):
            continue
        elif os.path.splitext(files)[1] in file_extensions:
            filelist.append(files)
    if absolute_dir:
        for i, file in enumerate(filelist):
            filelist[i] = root + filelist[i]

    return filelist


def deletespace(filename, target_dir):
    workbook = xlrd.open_workbook(filename, 'r')
    writebook = xlwt.Workbook()
    for i in range(len(workbook.sheet_names())):
        sheet = workbook.sheet_by_index(i)
        sheet1 = writebook.add_sheet('sheet' + str(i), cell_overwrite_ok=True)
        for j in range(len(sheet.col_values(1))):
            rows = sheet.row_values(j)
            rows[1] = rows[1].replace(' ', '')
            sheet1.write(j, 0, rows[0])
            sheet1.write(j, 1, rows[1])
            sheet1.write(j, 2, rows[2])
    writebook.save(target_dir + os.path.split(filename)[1])


# def searchfile(filename, target_dir):
#     if target_dir[-1] not in ['\\', '/']:
#         target_dir += ('/')
#     cur_dir_files = os.listdir(target_dir)
#     if filename in cur_dir_files:
#         return target_dir
#     for file in cur_dir_files:
#         if os.path.isdir(target_dir + file):
#             print(target_dir + file)
#             return searchfile(filename, target_dir + file)


def searchfile(filename, target_dir):
    for fpathe, dirs, fs in os.walk(target_dir):
        if filename in fs:
            return fpathe + '/'
    return ''


def main():
    # dataroot = 'D:\\Application\\dist\\韩国\\'
    # target_dir = 'D:\\Application\\excel'
    dataroot = input('输入需要复制的文件目录（仅复制当前目录下的文件）：')
    if not os.path.exists(dataroot):
        print(dataroot, ' input path does not exist!')
    target_dir = input('输入目标搜索文件目录')
    if not os.path.exists(target_dir):
        print(target_dir, ' input path does not exist!')
    filelist = GetFileList(dataroot, [
                           '.XLS', 'xls', 'XLSX', 'xlsx'], absolute_dir=False)
    print('file count: ', len(filelist))
    dataroot = Regular_Dir(dataroot)
    target_dir = Regular_Dir(target_dir)
    for i, file in enumerate(filelist):
        filedir = searchfile(file, target_dir)
        if not filedir:
            print(file + ' not found! copy failed!')
            continue
        if not os.path.exists(filedir + os.path.splitext(file)[0] + '_1' + os.path.splitext(file)[1]):
            os.rename(filedir + file, filedir + os.path.splitext(file)
                      [0] + '_1' + os.path.splitext(file)[1])
        else:
            print('err:' + filedir + os.path.splitext(file)
                  [0] + '_1' + os.path.splitext(file)[1] + ' exist, No more rename!,but copy is still execute')
            continue
        shutil.copyfile(dataroot + file, filedir + file)
        print('%d of %4d copy finish!' % (i + 1, len(filelist)))


main()
