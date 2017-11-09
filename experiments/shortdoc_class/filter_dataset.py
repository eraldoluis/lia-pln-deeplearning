# -*- coding: utf-8 -*-
# encoding=utf8
import sys
import re

#   Given an array of words (hashtags), this script will search through a dataset and keep the lines
#   in which at least one of the words in the array was found, in the following format:
#   <line>\t<words found separeted by space>

def keepIfFound(inFile, outFile, hashtags):
    with open(inFile) as fileSource, open(outFile, 'w+') as fileResult:
        countLines = 0
        countLinesKept = 0

        hashtags = [tag.lower() for tag in hashtags]

        #First line of result file shows the array of hashtags searched
        fileResult.write(hashtags.__str__()+"\n")

        for line in fileSource:
            countLines += 1
            resultLine = line[:-1].split("\t", 1)[1]

            tagsFound = []

            for token in resultLine.split():
                if token.startswith("#"):
                    token = token.lower()
                    for tag in hashtags:
                        if(token == tag and token not in tagsFound):
                            tagsFound.append(tag)
                            resultLine = re.compile(token, re.IGNORECASE).sub("##HASHTAG##", resultLine)
            if (len(tagsFound) > 0):
                tagsText = ""
                for tag in tagsFound:
                    tagsText += tag + " "
                fileResult.write(resultLine + "\t" + tagsText + "\n")
                countLinesKept += 1

    print "Result: " + str(countLinesKept) + " out of " + str(countLines) + " tweets were kept\nResult file: " + str(sys.argv[2])


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    keepIfFound(sys.argv[1],
                sys.argv[2],
                [
                    '#1',
                    '#umrei',
                    '#justin4mmva',
                    '#shawn4mmva',
                    '#nate:',
                    '#skype:',
                    '#onelovemanchester',
                    '#mtvawardsstar',
                    '#timbeta',
                    '#nowplaying',
                    '#premiosmtvmiaw',
                    '#masterchefbr',
                    '#네이버일반아이디판매',
                    '#네이버생성아이디판매',
                    '#트위터아이디판매',
                    '#구글아이디판매',
                    '#betaajudabeta',
                    '#peitudas',
                    '#bancodeseries',
                    '#sdv',
                    '#aforçadoquerer',
                    '#bomdia',
                    '#인스타그램아이디판매',
                    '#네이버휴먼아이디판매',
                    '#페이스북아이디판매',
                    '#skoob',
                    '#powercouplebrasil',
                    '#arianagrandenobrfebreteen',
                    '#rt',
                    '#sextadetremurasdv',
                    '#',
                    '#limeira',
                    '#sabadodetremurasdv',
                    '#edsheerannofebreteen',
                    '#2017btsfesta',
                    '#btsforvmas',
                    '#missaobeta',
                    '#veranomtv2017',
                    '#mtvexpopcd9',
                    '#mtvchallengecd9',
                    '#domingodetremurasdv',
                    '#pas',
                    '#2',
                    '#fjuamericana',
                    '#repost',
                    '#betalab',
                    '#harrystylesnanovela',
                    '#betaseguebeta',
                    '#teenchoice',
                    '#youngawardsmx17',
                    '#네이버아이디쪽지용판매',
                    '#페이스북아이디판매…',
                    '#네이버열심회원판매',
                    '#다음해킹아이디판매…',
                    '#페이스북해킹아이디판매',
                    '#구글생성아이디판매',
                    '#페이스북생성아이디판매',
                    '#검빛경마아이디판…',
                    '#다음해킹아이디판매',
                    '#보배드림아이디판매',
                    '#다음생성아이디판매',
                    '#일베아이디판매',
                    '#foratemer',
                    '#nbafinals',
                    '#hoytieneanittaparadinha',
                    '#quartadetremurasdv',
                    '#segundadetremurasdv',
                    '#quintadetremurasdv',
                    '#produce101final',
                    '#somostodosdilma',
                    '#np',
                    '#operacaobetalab',
                    '#queronotvz',
                    '#4yearswithbts',
                    '#malhação',
                    '#dança',
                    '#3',
                    '#forró',
                    '#xote',
                    '#forrozim',
                    '#pédeserra',
                    '#kpwww',
                    '#weathercloud',
                    '#raynniere',
                    '#lulanacadeia',
                    '#fabricadecasamentos',
                    '#brasil',
                    '#nbanaespn',
                    '#btshomeparty',
                    '#terçadetremurasdv',
                    '#e32017',
                    '#paz',
                    '#dancingbrasil',
                    '#bdsp',
                    '#diadosnamorados',
                    '#beta',
                    '#nãovounegar',
                    '#btsweek',
                    '#newtwitter',
                    '#5hdown',
                    '#4',
                    '#feriadodetremurasdv',
                    '#poramornoviva',
                    '#witness',
                    '#maislidas',
                    '#tercadetremurasdv',
                    '#noticias',
                    '#canaldaharu',
                    '#programadoporchat',
                    '#rockstory',
                    '#niall4mmva',
                    '#timbetaajudatimbeta',
                    '#soundcloud',
                    '#maisshow',
                    '#askbelieber',
                    '#rpsp',
                    '#emprego',
                    '#portugal',
                    '#photography',
                    '#sense8',
                    '#love',
                    '#mtvinstaglcabello',
                    '#5',
                    '#novomundo',
                    '#etsfs'
                ]
                )
