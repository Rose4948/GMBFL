# -*- coding: UTF-8 -*-
import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
from tqdm import tqdm
from scipy import sparse
import math
import json

#####摆烂的星期一

dmap = {
'Chart': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 24, 22: 25, 23: 26},
'Time': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,15: 17, 16: 18, 17: 19, 18: 20, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27},
'Math': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
             15: 106, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28, 27: 29,
             28: 30, 29: 31, 30: 32, 31: 33, 32: 34, 33: 35, 34: 36, 35: 37, 36: 38, 37: 39, 38: 40, 39: 41, 40: 42,
             41: 43, 42: 44, 43: 45, 44: 46, 45: 47, 46: 48, 47: 49, 48: 50, 49: 51, 50: 52, 51: 53, 52: 54, 53: 55,
             54: 56, 55: 57, 56: 58, 57: 59, 58: 60, 59: 61, 60: 62, 61: 63, 62: 64, 63: 65, 64: 66, 65: 67, 66: 68,
             67: 69, 68: 70, 69: 71, 70: 72, 71: 73, 72: 74, 73: 75, 74: 76, 75: 77, 76: 78, 77: 79, 78: 80, 79: 81,
             80: 82, 81: 83, 82: 84, 83: 85, 84: 86, 85: 87, 86: 88, 87: 89, 88: 90, 89: 91, 90: 92, 91: 93, 92: 94,
             93: 95, 94: 96, 95: 97, 96: 98, 97: 99, 98: 100, 99: 101, 100: 102, 101: 103, 102: 105},
'Cli':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 39, 6: 38, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19:20, 20: 21, 21:22, 22:23, 23:24,24: 25, 25:26, 26: 27,27: 28, 28:29,29:30,30: 31,31:32, 32: 33, 33: 34, 34: 35, 35: 37},
'Lang': {0: 1, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 24,22: 25,23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 51, 45: 52, 46: 53, 47: 54, 48: 55, 49: 57, 50: 58, 51: 59, 52: 60, 53: 61, 54: 62, 55: 63, 56: 64, 57: 65},
'JxPath': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22}
}



class SumDataset1( data.Dataset ):
    def __init__(self, config, dataName="train", proj="Math",testlst=[],vallst=[]):
        self.train_path = proj + ".pkl"
        self.val_path = "ndev.txt"
        self.test_path = "ntest.txt"
        self.proj = proj
        self.SentenceLen = config.SentenceLen
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Voc['Method'] = len( self.Nl_Voc )
        self.Nl_Voc['Test'] = len( self.Nl_Voc )
        self.Nl_Voc['Line'] = len( self.Nl_Voc )
        self.Nl_Voc['RTest'] = len( self.Nl_Voc )
        self.Nl_Voc['Mutation'] = len( self.Nl_Voc )
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.ids = []
        self.Nls = []
        self.firstFlog=False
        self.alldatalen=0
        if os.path.exists( "nl_voc.pkl" ):
            self.Load_Voc()
        else:
            self.init_dic()
        if self.firstFlog:
            data = pickle.load( open( self.proj + 'data.pkl', 'rb' ) )
        else:
            data = self.preProcessData( open( self.train_path, "rb" ) )
        self.firstFlog=True
        self.alldatalen = len(data[0])
        self.data = []
        if dataName == "train":
            for i in range( len( data ) ):
                tmp = []
                for j in range( len( data[i] ) ):
                    if j in testlst or j in vallst:
                        continue
                    tmp.append( data[i][j] )  # self.data.append(data[i][0:testid] + data[i][testid + 1:])
                self.data.append( tmp )
        elif dataName == 'test':
            self.ids = testlst
            for i in range( len( data ) ):
                tmp = []
                for x in self.ids:
                    tmp.append( data[i][x] )
                self.data.append( tmp )
        else:
            self.ids = vallst
            for i in range( len( data ) ):
                tmp = []
                for x in self.ids:
                    tmp.append( data[i][x] )
                self.data.append( tmp )

    def Load_Voc(self):
        if os.path.exists( "nl_voc.pkl" ):
            self.Nl_Voc = pickle.load( open( "nl_voc.pkl", "rb" ) )
            # print(self.Nl_Voc)
        if os.path.exists( "code_voc.pkl" ):
            self.Code_Voc = pickle.load( open( "code_voc.pkl", "rb" ) )
        if os.path.exists( "char_voc.pkl" ):
            self.Char_Voc = pickle.load( open( "char_voc.pkl", "rb" ) )

    def splitCamel(self, token):
        ans = []
        tmp = ""
        for i, x in enumerate( token ):
            if i != 0 and x.isupper() and token[i - 1].islower():
                ans.append( tmp )
                tmp = x.lower()
            elif x in '$.':
                ans.append( tmp )
                tmp = ""
            else:
                tmp += x.lower()
        ans.append( tmp )
        return ans

    def init_dic(self):
        print("initVoc")
        f = open( self.p + '.pkl', 'rb' )
        data = pickle.load( f )
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for x in data:
            for s in x['methods']:
                s = s[:s.index( '(' )]
                if len( s.split( ":" ) ) > 1:
                    tokens = ".".join( s.split( ":" )[0].split( '.' )[-2:] + [s.split( ":" )[1]] )
                else:
                    tokens = ".".join( s.split( ":" )[0].split( '.' )[-2:] )
                Codes.append( self.splitCamel( tokens ) )
                # print(Codes[-1])
            for s in x['ftest']:
                if len( s.split( ":" ) ) > 1:
                    tokens = ".".join( s.split( ":" )[0].split( '.' )[-2:] + [s.split( ":" )[1]] )
                else:
                    tokens = ".".join( s.split( ":" )[0].split( '.' )[-2:] )
                Codes.append( self.splitCamel( tokens ) )
        code_voc = VocabEntry.from_corpus( Codes, size=50000, freq_cutoff=0 )
        self.Code_Voc = code_voc.word2id
        open( "code_voc.pkl", "wb" ).write( pickle.dumps( self.Code_Voc ) )

    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            if x not in voc:
                ans.append( 1 )
            else:
                ans.append( voc[x] )
        return ans

    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append( c_id )
            ans.append( tmp )
        return ans

    def pad_seq(self, seq, maxlen):
        act_len = len( seq )
        if len( seq ) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_seq1(self, seq, maxlen):
        act_len = len( seq )
        if len(seq) < maxlen:
            seq = seq + [[self.PAD_token for _ in seq[0]]] * maxlen
            # print(seq[act_len:maxlen])
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_str_seq(self, seq, maxlen):
        act_len = len( seq )
        if len( seq ) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_list(self, seq, maxlen1, maxlen2):
        if len( seq ) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len( seq ) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def tokenize_for_bleu_eval(self, code):
        code = re.sub( r'([^A-Za-z0-9])', r' \1 ', code )
        # code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub( r'\s+', ' ', code )
        code = code.replace( '"', '`' )
        code = code.replace( '\'', '`' )
        tokens = [t for t in code.split( ' ' ) if t]
        return tokens

    def getoverlap(self, a, b):
        ans = []
        for x in a:
            maxl = 0
            for y in b:
                tmp = 0
                for xm in x:
                    if xm in y:
                        tmp += 1
                maxl = max( maxl, tmp )
            ans.append( int( 100 * maxl / len( x ) ) + 1 )
        return ans

    def getRes(self, codetoken, nltoken):
        ans = []
        for x in nltoken:
            if x == "<pad>":
                continue
            if x in codetoken and codetoken.index( x ) < self.Code_Len and x != "(" and x != ")":
                ans.append( len( self.Nl_Voc ) + codetoken.index( x ) )
            else:
                if x in self.Nl_Voc:
                    ans.append( self.Nl_Voc[x] )
                else:
                    ans.append( 1 )
        for x in ans:
            if x >= len( self.Nl_Voc ) + self.Code_Len:

                exit( 0 )
        return ans

    def preProcessData(self, dataFile):
        path_stacktrace = os.path.join( '../FLocalization/stacktrace', self.proj )
        # print(path_stacktrace)
        lines = pickle.load( dataFile )  # dataFile.readlines()
        Nodes = []
        Types = []
        LineNodes = []
        LineTypes = []
        LineMus = []
        Res = []
        inputText = []
        inputNlad = []
        LineSMType=[]
        maxl = 0
        maxl2 = 0
        error = 0
        error1 = 0
        error2 = 0
        error12 = 0
        correct = 0
        for k in range( len( lines ) ):
            x = lines[k]
            if os.path.exists( path_stacktrace + '/%d.json' % dmap[self.proj][k] ):
                # print("path_stacktrace:", path_stacktrace)
                stack_info = json.load( open( path_stacktrace + '/%d.json' % dmap[self.proj][k] ) )
                if x['ftest'].keys() != stack_info.keys():
                    with open( "problem_stack", 'a' ) as f:
                        f.write( "{} {} no!\n".format( k, dmap[self.proj][k] ) )
                        f.write( str( x['ftest'].keys() ) + '\n' )
                        f.write( str( stack_info.keys() ) + '\n' )
                    for error_trace in x['ftest'].keys():
                        if error_trace not in stack_info.keys():
                            error += 1
                        else:
                            correct += 1

            nodes = []
            types = []
            res = []
            nladrow = []
            nladcol = []
            nladval = []
            texta = []
            textb = []
            linenodes = []
            linetypes = []
            line_S_M =[]
            methodnum = len( x['methods'] )

            rrdict = {}

            for s in x['methods']:
                rrdict[x['methods'][s]] = s[:]

            for i in range( methodnum ):
                nodes.append( 'Method' )

                if len( rrdict[i].split("@") ) > 1:
                    tokens = ".".join( rrdict[i].split( "@" )[0].split( '/' )[-2:] + [rrdict[i].split( "@" )[1]] )
                else:
                    tokens = ".".join( rrdict[i].split( "@" )[0].split( '/' )[-2:] )
                ans = self.splitCamel( tokens )
                texta.append( ans )
                if i not in x['correctnum']:
                    types.append( 1 )
                else:
                    types.append( x['correctnum'][i] + 1 )
                if i in x['ans']:
                    res.append( 1 )
                else:
                    res.append( 0 )
            rrdic = {}
            for s in x['ftest']:
                rrdic[x['ftest'][s]] = s
            for i in range( len( x['ftest'] ) ):
                nodes.append( 'Test' )
                types.append( 0 )

                if len( rrdic[i].split( "#" ) ) > 1:
                    tokens = ".".join( rrdic[i].split( "#" )[0].split( '.' )[-2:] + [rrdic[i].split( "#" )[1]] )
                else:
                    tokens = ".".join( rrdic[i].split( "#" )[0].split( '.' )[-2:] )
                ans = self.splitCamel( tokens )
                textb.append( ans )
            for i in range( len( x['rtest'] ) ):
                nodes.append( 'RTest' )
                types.append( 0 )

            mus = []
            for i in range( len( x['lines'] ) ):
                if i not in x['ltype']:
                    x['ltype'][i] = 'Empty'
                if x['ltype'][i] not in self.Nl_Voc:
                    self.Nl_Voc[x['ltype'][i]] = len( self.Nl_Voc )
                linenodes.append( x['ltype'][i] )
                if i in x['lcorrectnum']:
                    linetypes.append( x['lcorrectnum'][i] )
                else:
                    linetypes.append( 1 )
                line_S_M.append(x['lSBFL'][i])

            for i in range( len( x['mutation'] ) ):
                if i not in x['mtype']:
                    x['mtype'][i] = 'Empty'
                if x['mtype'][i] not in self.Nl_Voc:
                    self.Nl_Voc[x['mtype'][i]] = len( self.Nl_Voc )
                linenodes.append( x['mtype'][i] )
                linetypes.append( 0 )
                line_S_M.append( x['mMBFL'][i] )
            maxl = max( maxl, len( nodes ) )
            maxl2 = max( maxl2, len( linenodes ) )
            print(maxl, maxl2)
            ed = {}

            line2method = {}
            for e in x['edge2']:
                line2method[e[1]] = e[0]
                a = e[0]
                b = e[1] + self.Nl_Len
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    assert (0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    assert (0)
                nladrow.append( a )
                nladcol.append( b )
                nladval.append( 1 )
                nladrow.append( b )
                nladcol.append( a )
                nladval.append( 1 )
            for e in x['edge10']:
                if e[0] not in line2method:
                    error1 += 1
                a = e[0] + self.Nl_Len
                b = e[1] + methodnum + len( x['ftest'] )
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                    nladrow.append( a )
                    nladcol.append( b )
                    nladval.append( 1 )
                else:
                    pass
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                    nladrow.append( b )
                    nladcol.append( a )
                    nladval.append( 1 )
                else:
                    pass

            for e in x['edge']:
                if e[0] not in line2method:
                    error2 += 1
                a = e[0] + self.Nl_Len
                b = e[1] + methodnum
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                    nladrow.append( a )
                    nladcol.append( b )
                    nladval.append( 1 )
                else:
                    assert (0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                    nladrow.append( b )
                    nladcol.append( a )
                    nladval.append( 1 )
                else:
                    assert (0)

            for e in x['edge12']:
                if e[1] not in line2method:
                    error12 += 1
                a = e[0] + self.Nl_Len + len( x['lines'] )
                b = e[1] + self.Nl_Len

                if (a, b) not in ed:
                    ed[(a, b)] = 1
                    nladrow.append( a )
                    nladcol.append( b )
                    nladval.append( 1 )
                else:
                    assert (0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                    nladrow.append( b )
                    nladcol.append( a )
                    nladval.append( 1 )
                else:
                    assert (0)

            for e in x['edge13']:
                a = e[0] + self.Nl_Len + len( x['lines'] )
                b = e[1] + len( x['ftest'] ) + methodnum

                if (a, b) not in ed:
                    ed[(a, b)] = 1
                    nladrow.append( a )
                    nladcol.append( b )
                    nladval.append( 1 )
                else:
                    assert (0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                    nladrow.append( b )
                    nladcol.append( a )
                    nladval.append( 1 )
                else:
                    assert (0)

            for e in x['edge14']:
                a = e[0] + self.Nl_Len + len( x['lines'] )
                b = e[1] + methodnum

                if (a, b) not in ed:
                    ed[(a, b)] = 1
                    nladrow.append( a )
                    nladcol.append( b )
                    nladval.append( 1 )
                else:
                    assert (0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                    nladrow.append( b )
                    nladcol.append( a )
                    nladval.append( 1 )
                else:
                    assert (0)

            overlap = self.getoverlap( texta, textb )

            Nodes.append( self.pad_seq( self.Get_Em( nodes, self.Nl_Voc ), self.Nl_Len ) )
            Types.append( self.pad_seq( types, self.Nl_Len ) )
            Res.append( self.pad_seq( res, self.Nl_Len ) )
            LineMus.append( self.pad_list( mus, self.Code_Len, 3 ) )
            inputText.append( self.pad_seq( overlap, self.Nl_Len ) )
            LineNodes.append( self.pad_seq( self.Get_Em( linenodes, self.Nl_Voc ), self.Code_Len ) )
            LineTypes.append( self.pad_seq( linetypes, self.Code_Len ) )
            LineSMType.append(self.pad_seq1( line_S_M, self.Code_Len ) )
            row = {}
            col = {}
            for i in range( len( nladrow ) ):
                if nladrow[i] not in row:
                    row[nladrow[i]] = 0
                row[nladrow[i]] += 1
                if nladcol[i] not in col:
                    col[nladcol[i]] = 0
                col[nladcol[i]] += 1
            for i in range( len( nladrow ) ):
                nladval[i] = 1 / math.sqrt( row[nladrow[i]] ) * 1 / math.sqrt( col[nladcol[i]] )
            nlad = sparse.coo_matrix( (nladval, (nladrow, nladcol)),
                                      shape=(self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len) )
            inputNlad.append( nlad )
        print("max1: %d max2: %d" % (maxl, maxl2))
        print("correct: %d error: %d" % (correct, error))
        print("error1: %d error2: %d" % (error1, error2))


        batchs = [Nodes, Types, inputNlad, Res, inputText, LineNodes, LineTypes, LineMus,LineSMType]
        self.data = batchs
        open( self.proj + "data.pkl", "wb" ).write( pickle.dumps( batchs, protocol=4 ) )
        return batchs

    def __getitem__(self, offset):
        ans = []
        if True:
            for i in range( len( self.data ) ):
                if i == 2:
                    ans.append( self.data[i][offset].toarray() )
                else:
                    ans.append( np.array( self.data[i][offset] ) )
        else:
            for i in range( len( self.data ) ):
                if i == 4:
                    continue
                ans.append( np.array( self.data[i][offset] ) )
            negoffset = random.randint( 0, len( self.data[0] ) - 1 )
            while negoffset == offset:
                negoffset = random.randint( 0, len( self.data[0] ) - 1 )
            if self.dataName == "train":
                ans.append( np.array( self.data[2][negoffset] ) )
                ans.append( np.array( self.data[3][negoffset] ) )
        return ans

    def __len__(self):
        return len( self.data[0] )

    def Get_Train(self, batch_size):
        data = self.data
        loaddata = data
        batch_nums=0
        if batch_size!=0:
            batch_nums = int( len( data[0] ) / batch_size )
        if True:
            if self.dataName == 'train':
                shuffle = np.random.permutation( range( len( loaddata[0] ) ) )  # 把数据长度的列表打乱
            else:
                shuffle = np.arange( len( loaddata[0] ) )
            for i in range(batch_nums):
                ans = []
                for j in range( len( data ) ):
                    if j != 2:
                        tmpd = np.array( data[j] )[shuffle[batch_size * i: batch_size * (i + 1)]]
                        ans.append( torch.from_numpy( np.array( tmpd ) ) )  # 把数组转换成张量，二者共享内存
                    else:
                        ids = []
                        v = []
                        for idx in range( batch_size * i, batch_size * (i + 1) ):
                            for p in range( len( data[j][shuffle[idx]].row ) ):
                                ids.append(
                                    [idx - batch_size * i, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]] )
                                v.append( data[j][shuffle[idx]].data[p] )
                        ans.append( torch.sparse.FloatTensor( torch.LongTensor(ids).t(), torch.FloatTensor( v ),
torch.Size([batch_size, self.Nl_Len + self.Code_Len,
self.Nl_Len + self.Code_Len] ) ) )
                yield ans
            if batch_nums * batch_size < len( data[0] ):
                ans = []
                for j in range( len( data ) ):
                    if j != 2:
                        tmpd = np.array( data[j] )[shuffle[batch_nums * batch_size:]]
                        ans.append( torch.from_numpy( np.array( tmpd ) ) )
                    else:
                        ids = []
                        v = []
                        for idx in range( batch_size * batch_nums, len( data[0] ) ):
                            for p in range( len( data[j][shuffle[idx]].row ) ):
                                ids.append( [idx - batch_size * batch_nums, data[j][shuffle[idx]].row[p],
                                             data[j][shuffle[idx]].col[p]] )
                                v.append( data[j][shuffle[idx]].data[p] )
                        ans.append( torch.sparse.FloatTensor( torch.LongTensor( ids ).t(), torch.FloatTensor( v ),
                                                              torch.Size( [len( data[0] ) - batch_size * batch_nums,
                                                                           self.Nl_Len + self.Code_Len,
                                                                           self.Nl_Len + self.Code_Len] ) ) )
                yield ans


class node:
    def __init__(self, name):
        self.name = "Lang"
        self.father = None
        self.child = []
        self.id = -1
