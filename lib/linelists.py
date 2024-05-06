import re
import sys
import logging
import pandas as pd
from pysme.linelist.linelist import LineList

logger = logging.getLogger(__name__)

def readlinelistfile(linelistfilename, segments=None):

    logger.info("Loading line list: "+linelistfilename)
    f = open(linelistfilename, "r")

    expectedlines = 0
    # look for headers
    linecountmatch = re.compile(r"\s*\d+\.?\d*,\s+\d+\.?\d*,\s+(\d+),")
    headerdatamatch = re.compile(r"\s*(Elm|Spec) Ion\s+WL_air\(A\)\s+log gf\*\s+E_low\(eV\)\s+J lo\s+E_up\(eV\)\s+J up\s+lower\s+upper\s+mean\s+Rad.\s+Stark\s+Waals")
    while True:
        fileline = f.readline()
        if len(fileline) == 0:
            logger.error("line list not LONG format? unable to locate headers in line list: "+linelistfilename)
            f.close()
            sys.exit(1)
        result = linecountmatch.search(fileline)
        if result:
            expectedlines = int(result.group(1))
        if headerdatamatch.search(fileline):
            break

    # data structure to contain line list
    data = {
        "species": [],
        "wlcent": [],
        "gflog": [],
        "excit": [],
        "j_lo": [],
        "e_upp": [],
        "j_up": [],
        "lande_lower": [],
        "lande_upper": [],
        "lande": [],
        "gamrad": [],
        "gamqst": [],
        "gamvw": [],
        "term_lower": [],
        "term_upper": [],
        "reference": [],
        "error": []
    }

    counttotal = 0
    countincl = 0
    # read line data
    quotematch = re.compile(r"\s*'?([^']*)'?")
    while True:
        dataline = f.readline().rstrip('\r\n')
        term1line = f.readline().rstrip('\r\n')
        term2line = f.readline().rstrip('\r\n')
        refline = f.readline().rstrip('\r\n')
        if len(fileline) == 0 or len(term1line) == 0 or len(term2line) == 0 or len(refline) == 0: break

        dataelems = dataline.split(',')

        if len(dataelems)<13:
            break

        counttotal += 1

        wavelength = float(dataelems[1])
        if segments is not None:
            lineinsegment = False
            for start, end in segments:
                if start <= wavelength <= end:
                    lineinsegment = True
            if not lineinsegment: continue

        result = quotematch.search(dataelems[0])
        if result:
            elmion = result.group(1)
        else:
            logger.error('unable to parse Elm Ion in list entry:\n'+dataline+term1line+term2line+refline)
            continue
        result = quotematch.search(term1line)
        if result:
            term1 = re.sub(r"\s+", ' ', result.group(1)[8:].strip())  # Ignore if its LS / JK / Hb / etc.
        else:
            logger.error('unable to parse Term1 in list entry:\n'+dataline+term1line+term2line+refline)
            continue
        result = quotematch.search(term2line)
        if result:
            term2 = re.sub(r"\s+", ' ', result.group(1)[8:].strip())  # Ignore if its LS / JK / Hb / etc.
        else:
            logger.error('unable to parse Term2 in list entry:\n'+dataline+term1line+term2line+refline)
            continue
        result = quotematch.search(refline)
        if result:
            ref = re.sub(r"\s+", ' ', result.group(1).strip())
        else:
            logger.error('unable to parse reference line in list entry:\n'+dataline+term1line+term2line+refline)
            continue

        countincl += 1
        data["species"].append(elmion)
        data["wlcent"].append(wavelength)
        data["gflog"].append(float(dataelems[2]))
        data["excit"].append(float(dataelems[3]))
        data["j_lo"].append(float(dataelems[4]))
        data["e_upp"].append(float(dataelems[5]))
        data["j_up"].append(float(dataelems[6]))
        data["lande_lower"].append(float(dataelems[7]))
        data["lande_upper"].append(float(dataelems[8]))
        data["lande"].append(float(dataelems[9]))
        data["gamrad"].append(float(dataelems[10]))
        data["gamqst"].append(float(dataelems[11]))
        data["gamvw"].append(float(dataelems[12]))
        data["term_lower"].append(term1)
        data["term_upper"].append(term2)
        data["reference"].append(ref)
        data["error"].append(0.5)

    f.close()
    if counttotal != expectedlines:
        logger.warning("expected to find "+str(expectedlines)+" but found "+str(counttotal))
    logger.info("Loaded "+str(countincl)+" lines of "+str(counttotal)+" (limiting to segments:"+str(segments is not None)+")")
    linedata = pd.DataFrame.from_dict(data)
    return linedata, "long"


#short line list reader really hacky to be backward compatible with old scripts, stick to long format please
def readlinelistfileshort(linelistfilename, segments=None):

    logger.info("Loading line list: "+linelistfilename)
    f = open(linelistfilename, "r")

    expectedlines = 0
    # look for headers
    linecountmatch = re.compile(r"\s*\d+\.?\d*,\s+\d+\.?\d*,\s+(\d+),")
    headerdatamatch1 = re.compile(r"\s*Spec Ion\s+WL\(A\)\s+Excit\(eV\)\s+Vmic\s+log\(gf\)\s+Rad\.\s+Stark\s+Waals\s+factor\s+depth\s+Reference")
    headerdatamatch2 = re.compile(r"\s*Elm Ion\s+WL_air\(A\)\s+Excit\(eV\)\s+log gf\*\s+Rad\.\s+Stark\s+Waals\s+factor\s+Reference")
    while True:
        fileline = f.readline()
        if len(fileline) == 0:
            logger.error("line list not SHORT format? unable to locate headers in line list: "+linelistfilename)
            f.close()
            sys.exit(1)
        result = linecountmatch.search(fileline)
        if result:
            expectedlines = int(result.group(1))
        if headerdatamatch1.search(fileline) or headerdatamatch2.search(fileline):
            break

    # data structure to contain line list
    data = {
        "species": [],
        "wlcent": [],
        "excit": [],
        "gflog": [],
        "vmic": [],
        "gamrad": [],
        "gamqst": [],
        "gamvw": [],
        "lande": [],
        "reference": [],
        "error": [],
    }
    ionizations = []

    counttotal = 0
    countincl = 0
    # read line data
    noionquotematch = re.compile(r"\s*'(\w+)'")
    quotematch = re.compile(r"\s*'([^']*)'")
    lastquotematch = re.compile(r"\s*'([^']*)'$")
    while True:
        dataline = f.readline().rstrip()
        if len(fileline) == 0: break

        dataelems = dataline.split(',')

        if len(dataelems)<10:
            break

        counttotal += 1

        wavelength = float(dataelems[1])
        if segments is not None:
            lineinsegment = False
            for start, end in segments:
                if start <= wavelength <= end:
                    lineinsegment = True
            if not lineinsegment: continue

        result = noionquotematch.search(dataelems[0])
        ionization = -1
        if result:
            elmion = result.group(1)
            ionization = 1
        else:
            result = quotematch.search(dataelems[0])
            if result:
                elmion = result.group(1)
            else:
                logger.error('unable to parse Elm Ion in list entry:\n'+dataline)
                continue
        result = lastquotematch.search(dataline)
        if result:
            ref = re.sub(r"\s+", ' ', result.group(1))
        else:
            logger.error('unable to parse reference field in list entry:\n'+dataline)
            continue

        countincl += 1
        data["species"].append(elmion)
        data["wlcent"].append(wavelength)
        data["excit"].append(float(dataelems[2]))
        data["vmic"].append(float(dataelems[3]))
        data["gflog"].append(float(dataelems[4]))
        data["gamrad"].append(float(dataelems[5]))
        data["gamqst"].append(float(dataelems[6]))
        data["gamvw"].append(float(dataelems[7]))
        data["lande"].append(float(dataelems[8]))
        data["reference"].append(ref)
        data["error"].append(0.5)
        if ionization>-1: ionizations.append(ionization)

    f.close()
    if len(ionizations)>0:
        if len(ionizations)==len(data["species"]):
            data["ionization"] = ionizations
        else:
            logger.error("some species have ionization nubmers and others do not, most be consistent in file: "+linelistfilename)
            sys.exit(1)
    if counttotal != expectedlines:
        logger.warning("expected to find "+str(expectedlines)+" but found "+str(counttotal))
    logger.info("Loaded "+str(countincl)+" lines of "+str(counttotal)+" (limiting to segments:"+str(segments is not None)+")")
    linedata = pd.DataFrame.from_dict(data)
    return linedata, "short"


def mergelinelists(linelistfilenames, segments=None, droplinelistduplicates=False, fileformat="long"):

    linedatas = []
    for linelistfilename in linelistfilenames:
        if fileformat== "short":
            linedata, returnedlineformat = readlinelistfileshort(linelistfilename, segments)
            if returnedlineformat != "short":
                logger.error('was expecting short format for line list: '+linelistfilename)
                sys.exit(1)
            linedatas.append(linedata)
        elif fileformat=="long":
            linedata, returnedlineformat = readlinelistfile(linelistfilename, segments)
            if returnedlineformat != "long":
                logger.error('was expecting long format for line list: ' + linelistfilename)
                sys.exit(1)
            linedatas.append(linedata)
        else:
            logger.error('was expecting long or short format for line list: ' + linelistfilename)
            sys.exit(1)
    concatlinedata = pd.concat(linedatas).sort_values(by="wlcent").reset_index(drop=True)

    if droplinelistduplicates:
        lastspecies = concatlinedata['species'].iloc[0]
        lastwlcent = concatlinedata['wlcent'].iloc[0]
        lastgflog = concatlinedata['gflog'].iloc[0]
        lastgamvw = concatlinedata['gamvw'].iloc[0]
        for idx, row in concatlinedata.iterrows():
            if idx == 0:
                continue
            if lastspecies == row['species'] and abs(lastwlcent - row['wlcent']) < 0.003 and abs(lastgflog - row['gflog']) < 0.003 and abs(lastgamvw - row['gamvw']) < 0.003:
                logger.warning("Duplicate removed: "+lastspecies+" wave:"+str(lastwlcent)+"#"+str(row['wlcent'])+" log(gf):"+str(lastgflog)+"#"+str(row['gflog'])+" VdW:"+str(lastgamvw)+"#"+str(row['gamvw']))
                concatlinedata.drop(idx, inplace=True)
            lastspecies = row['species']
            lastwlcent = row['wlcent']
            lastgflog = row['gflog']
            lastgamvw = row['gamvw']
        concatlinedata = concatlinedata.reset_index(drop=True)

    return LineList(concatlinedata, fileformat)
