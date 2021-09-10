import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
plt.ion()
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# ========== Maximum Source Density Bins ===============================
# ======================================================================
# fig1
plt.figure(figsize=(10,7))
plt.xlabel('Source density / 5\"-pixel', fontsize=16)
plt.ylabel('N', fontsize=16)
plt.title('Max SD > 20 sources/pixel')
# fig2
plt.figure(figsize=(10,7))
plt.xlabel('Source density / 5\"-pixel', fontsize=16)
plt.ylabel('N', fontsize=16)
plt.title('3 < Max SD <= 5 sources/pixel')
# fig3
plt.figure(figsize=(10,7))
plt.xlabel('Source density / 5\"-pixel', fontsize=16)
plt.ylabel('N', fontsize=16)
plt.title('2 < Max SD <= 3 sources/pixel')
# fig4
plt.figure(figsize=(10,7))
plt.xlabel('Source density / 5\"-pixel', fontsize=16)
plt.ylabel('N', fontsize=16)
plt.title('1 < Max SD <= 2 sources/pixel')
# fig5
plt.figure(figsize=(10,7))
plt.xlabel('Source density / 5\"-pixel', fontsize=16)
plt.ylabel('N', fontsize=16)
plt.title('Max SD <= 1 sources/pixel')

means = []
maxes = []

for i in range(len(files)):
    t = Table.read(files[i])
    srcden = t['SourceDensity']
    n = plt.hist(srcden, bins=15, histtype='step', lw=0.1, color='w');
    maxn = np.around(max(n[1]), decimals=2)
    meansd = np.around(np.mean(srcden), decimals=3)
    means.append(meansd)
    maxes.append(maxn)
    print('max n[1]', max(n[1]))
    if max(n[1]) > 20:
        print('SD > 20')
        plt.figure(1)
        plt.hist(srcden, bins=15, histtype='step', lw=1.5, label='%s; max=%s; mean=%s' % (files[i].split('M31-')[1].split('.')[0], maxn, meansd))
        # plt.axvline(meansd, max(n[0]), c='gray')
    if (max(n[1]) > 3 and max(n[1]) <= 5):
        print('3 < SD <= 5')
        plt.figure(2)
        plt.hist(srcden, bins=15, histtype='step', lw=1.5, label='%s; max=%s; mean=%s' % (files[i].split('M31-')[1].split('.')[0], maxn, meansd))
        # plt.axvline(meansd, max(n[0]), c='gray')
    if (max(n[1]) > 2 and max(n[1]) < 3):
        print('3 < SD <= 5')
        plt.figure(3)
        plt.hist(srcden, bins=15, histtype='step', lw=1.5, label='%s; max=%s; mean=%s' % (files[i].split('M31-')[1].split('.')[0], maxn, meansd))
        # plt.axvline(meansd, max(n[0]), c='gray')
    if (max(n[1]) > 1 and max(n[1]) <= 2):
        print('1 < SD <= 3')
        plt.figure(4)
        plt.hist(srcden, bins=15, histtype='step', lw=1.5, label='%s; max=%s; mean=%s' % (files[i].split('M31-')[1].split('.')[0], maxn, meansd))
        # plt.axvline(meansd, max(n[0]), c='gray')
    if max(n[1]) <= 1:
        print('SD <= 1')
        plt.figure(5)
        plt.hist(srcden, bins=15, histtype='step', lw=1.5,
                 label='%s; max=%s; mean=%s' % (files[i].split('M31-')[1].split('.')[0], maxn, meansd)
                 )
        # plt.axvline(meansd, max(n[0]), c='gray')

    # print('\n', i, '\n', n[1], '\n', max(n[1]))
    print(i)
ax = plt.gca()
ymax = ax.get_ylim()[1]
xmax = ax.get_xlim()[1]
plt.xlim(-1.15, xmax+1)
plt.ylim(-10000, ymax)
ax.xaxis.set_minor_locator(MultipleLocator(1))
plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()
plt.figure(3)
plt.legend(ncol=2)
plt.figure(4)
plt.legend(ncol=3)
plt.figure(5)
plt.legend(ncol=3)


# =================== Mean Source Density Bins ========================
# =====================================================================
# fig1
plt.figure(figsize=(10,7))
plt.xlabel('Source density / 5\"-pixel', fontsize=16)
plt.ylabel('N', fontsize=16)
plt.title('Mean SD > 0.4 sources/pixel')
# fig2
plt.figure(figsize=(10,7))
plt.xlabel('Source density / 5\"-pixel', fontsize=16)
plt.ylabel('N', fontsize=16)
plt.title('0.2 < Mean SD <= 0.4 sources/pixel')
# fig3
plt.figure(figsize=(10,7))
plt.xlabel('Source density / 5\"-pixel', fontsize=16)
plt.ylabel('N', fontsize=16)
plt.title('0.15 < Mean SD <= 0.2 sources/pixel')
# fig4
plt.figure(figsize=(10,7))
plt.xlabel('Source density / 5\"-pixel', fontsize=16)
plt.ylabel('N', fontsize=16)
plt.title('0.1 < Mean SD <= 0.15 sources/pixel')
# fig5
plt.figure(figsize=(10,7))
plt.xlabel('Source density / 5\"-pixel', fontsize=16)
plt.ylabel('N', fontsize=16)
plt.title('Mean SD <= 0.1 sources/pixel')

means2 = []
maxes2 = []

for i in range(len(files)):
    t = Table.read(files[i])
    srcden = t['SourceDensity']
    n = plt.hist(srcden, bins=15, histtype='step', lw=0.1, color='w');
    maxn = np.around(max(n[1]), decimals=2)
    meansd = np.around(np.mean(srcden), decimals=3)
    means2.append(meansd)
    maxes2.append(maxn)
    if meansd > 0.4:
        print('SD > 0.4')
        plt.figure(1)
        plt.hist(srcden, bins=15, histtype='step', lw=1.5, label='%s; max=%s; mean=%s' % (files[i].split('M31-')[1].split('.')[0], maxn, meansd))
    if (meansd > 0.2 and meansd <= 0.4):
        print('0.2 < SD <= 0.4')
        plt.figure(2)
        plt.hist(srcden, bins=15, histtype='step', lw=1.5, label='%s; max=%s; mean=%s' % (files[i].split('M31-')[1].split('.')[0], maxn, meansd))
    if (meansd > 0.15 and meansd < 0.2):
        print('0.15 < SD <= 0.2')
        plt.figure(3)
        plt.hist(srcden, bins=15, histtype='step', lw=1.5, label='%s; max=%s; mean=%s' % (files[i].split('M31-')[1].split('.')[0], maxn, meansd))
    if (meansd > 0.1 and meansd <= 0.15):
        print('0.1 < SD <= 0.15')
        plt.figure(4)
        plt.hist(srcden, bins=15, histtype='step', lw=1.5, label='%s; max=%s; mean=%s' % (files[i].split('M31-')[1].split('.')[0], maxn, meansd))
    if meansd <= 0.1:
        print('SD <= 0.1')
        plt.figure(5)
        plt.hist(srcden, bins=15, histtype='step', lw=1.5, label='%s; max=%s; mean=%s' % (files[i].split('M31-')[1].split('.')[0], maxn, meansd))
    print(i)

ax = plt.gca()
ymax = ax.get_ylim()[1]
xmax = ax.get_xlim()[1]
plt.xlim(-0.1, xmax+0.1)
plt.ylim(-10000, ymax)
ax.xaxis.set_minor_locator(MultipleLocator(1))
plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()
plt.figure(3)
plt.legend()
plt.figure(4)
plt.legend(ncol=2)
plt.figure(5)
plt.legend(ncol=2)


# ==================== Means & Maximums per half brick ========================
# =============================================================================


flds = ['1E','1W','2E','2W','3E','3W','4E','4W','5E','5W','6E','6W','7E'
      ,'7W','8E','8W','9E','9W','10E','10W','11E','11W','12E','12W','13E','13W',
        '14E','14W','15E','15W','16E','16W','17E','17W','18E','18W','19E','19W',
        '20E','20W','21E','21W','22E','22W','23E']

plt.figure()
plt.title('Mean source density per field', fontsize=16)
plt.plot(flds, means, 'o', ls='')
plt.xlabel('Brick', fontsize=16)
plt.ylabel('Mean source density', fontsize=16)

plt.figure()
plt.title('Maximum source density per field', fontsize=16)
plt.plot(flds, maxes, 'o', color='tomato', ls='')
plt.xlabel('Brick', fontsize=16)
plt.ylabel('Max source density', fontsize=16)