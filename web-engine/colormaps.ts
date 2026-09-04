/**
 * Colormap lookup tables, for OpenCV builds that omit `cv.applyColorMap`
 * (confirmed absent from the opencv-wasm@4.3.0-10 imgproc bindings this project
 * loads — see opencv.ts).
 *
 * These are OpenCV's own tables, not approximations of them: each is the exact
 * 256-entry ramp `cv2.applyColorMap` produces, extracted once and baked in, so
 * a heatmap rendered here matches the desktop pixel for pixel. Hand-fitted
 * curves came within ~50 levels on the perceptual maps and were plainly wrong
 * on Rainbow, which is why they were replaced.
 *
 * Each table is 768 bytes (256 x RGB) carried as base64 — about 1 kB of source
 * per map, decoded once on first use.
 */

/** Decodes one base64 table into a flat Uint8Array of 256 RGB triples. */
function decodeLut(encoded: string): Uint8Array {
  const binary = atob(encoded)
  const out = new Uint8Array(768)
  for (let i = 0; i < 768; i++) out[i] = binary.charCodeAt(i)
  return out
}

/** Wraps a table as the (v) => [r, g, b] lookup the node code calls. */
function lutColor(encoded: string): (v: number) => [number, number, number] {
  let table: Uint8Array | null = null
  return (v: number) => {
    if (!table) table = decodeLut(encoded)
    const i = (Math.max(0, Math.min(255, Math.round(v))) | 0) * 3
    return [table[i], table[i + 1], table[i + 2]]
  }
}

const JET_LUT = 'AACAAACEAACIAACMAACQAACUAACYAACcAACgAACkAACoAACsAACwAAC0AAC4AAC8AADAAADEAADIAADMAADQAADUAADYAADcAADgAADkAADoAADsAADwAAD0AAD4AAD8AAD/AAT/AAj/AAz/ABD/ABT/ABj/ABz/ACD/ACT/ACj/ACz/ADD/ADT/ADj/ADz/AED/AET/AEj/AEz/AFD/AFT/AFj/AFz/AGD/AGT/AGj/AGz/AHD/AHT/AHj/AHz/AID/AIT/AIj/AIz/AJD/AJT/AJj/AJz/AKD/AKT/AKj/AKz/ALD/ALT/ALj/ALz/AMD/AMT/AMj/AMz/AND/ANT/ANj/ANz/AOD/AOT/AOj/AOz/APD/APT/APj/APz/Av/+Bv/6Cv/2Dv/yEv/uFv/qGv/mHv/iIv/eJv/aKv/WLv/SMv/ONv/KOv/GPv/CQv++Rv+6Sv+2Tv+yUv+uVv+qWv+mXv+iYv+eZv+aav+Wbv+Scv+Odv+Kev+Gfv+Cgv9+hv96iv92jv9ykv9ulv9qmv9mnv9iov9epv9aqv9Wrv9Ssv9Otv9Kuv9Gvv9Cwv8+xv86yv82zv8y0v8u1v8q2v8m3v8i4v8e5v8a6v8W7v8S8v8O9v8K+v8G/v8B//wA//gA//QA//AA/+wA/+gA/+QA/+AA/9wA/9gA/9QA/9AA/8wA/8gA/8QA/8AA/7wA/7gA/7QA/7AA/6wA/6gA/6QA/6AA/5wA/5gA/5QA/5AA/4wA/4gA/4QA/4AA/3wA/3gA/3QA/3AA/2wA/2gA/2QA/2AA/1wA/1gA/1QA/1AA/0wA/0gA/0QA/0AA/zwA/zgA/zQA/zAA/ywA/ygA/yQA/yAA/xwA/xgA/xQA/xAA/wwA/wgA/wQA/wAA/AAA+AAA9AAA8AAA7AAA6AAA5AAA4AAA3AAA2AAA1AAA0AAAzAAAyAAAxAAAwAAAvAAAuAAAtAAAsAAArAAAqAAApAAAoAAAnAAAmAAAlAAAkAAAjAAAiAAAhAAAgAAA'
const VIRIDIS_LUT = 'RAFURAJWRQRXRQVZRgdaRghcRgpdRgteRw1gRw5hRxBjRxFkRxNlSBRnSBZoSBdpSBhqSBpsSBttSBxuSB1vSB9wSCBxSCFzSCN0SCR1SCV2SCZ3SCh4SCl5Ryp6Ryx6Ry17Ry58Ry99RjB+RjJ+RjN/RjSARTWBRTeBRTiCRDmDRDqDRDuEQz2EQz6FQj+FQkCGQkGGQUKHQUSHQEWIQEaIP0eIP0iJPkmJPkqJPkyKPU2KPU6KPE+KPFCLO1GLO1KLOlOLOlSMOVWMOVaMOFiMOFmMN1qMN1uNNlyNNl2NNV6NNV+NNGCNNGGNM2KNM2ONMmSOMmWOMWaOMWeOMWiOMGmOMGqOL2uOL2yOLm2OLm6OLm+OLXCOLXGOLHGOLHKOLHOOK3SOK3WOKnaOKneOKniOKXmOKXqOKXuOKHyOKH2OJ36OJ3+OJ4COJoGOJoKOJoKOJYOOJYSOJYWOJIaOJIeOI4iOI4mOI4qNIouNIoyNIo2NIY6NIY+NIZCNIZGMIJKMIJKMIJOMH5SMH5WLH5aLH5eLH5iLH5mKH5qKHpuKHpyJHp2JH56JH5+IH6CIH6GIH6GHH6KHIKOGIKSGIaWFIaaFIqeFIqiEI6mDJKqDJauCJayCJq2BJ62BKK6AKa9/KrB/LLF+LbJ9LrN8L7R8MbV7MrZ6NLZ5Nbd5N7h4OLl3Orp2O7t1Pbx0P7xzQL1yQr5xRL9wRsBvSMFuSsFtTMJsTsNrUMRqUsVpVMVoVsZnWMdlWshkXMhjXsliYMpgY8tfZcteZ8xcac1bbM1abs5YcM9Xc9BWddBUd9FTetFRfNJQf9NOgdNNhNRLhtVJidVIi9ZGjtZFkNdDk9dBldhAmNg+m9k8ndk7oNo5oto3pds2qNs0qtwyrdwwsN0vst0ttd4ruN4put4ovd8mwN8lwt8jxeAhyOAgyuEfzeEd0OEc0uIb1eIa2OIZ2uMZ3eMY3+MY4uQY5eQZ5+QZ6uUa7OUb7+Uc8eUd9OYe9uYg+OYh++cj/ecl'
const PLASMA_LUT = 'DQiHEAeIEweJFgeKGQaMGwaNHQaOIAaPIgaQJAaRJgWRKAWSKgWTLAWULgWVLwWWMQWXMwWXNQSYNwSZOASaOgSaPASbPgScPwScQQSdQwOeRAOeRgOfSAOfSQOgSwOhTAKhTgKiUAKiUQKjUwKjVQKkVgGkWAGkWQGlWwGlXAGmXgGmYAGmYQCnYwCnZACnZgCnZwCoaQCoagCobACobgCobwCocQCocgGodAGodQGodwGoeAGoegKoewKofQOofgOogASogQSngwWnhAWnhgamhwemiAimigmliwqljQuljgykjw2kkQ6jkg+jlBCilRGhlhOhmBSgmRWfmhafnBeenRidnhmdoBqcoRuboh2aox6apR+ZpiCYpyGXqCKWqiOVqySUrCaUrSeTriiSsCmRsSqQsiuPsyyOtC6NtS+MtjCLtzGKuDKJujOIuzSIvDWHvTeGvjiFvzmEwDqDwTuCwjyBwz2AxD5/xUB+xkF9x0J8yEN7yUR6ykV6y0Z5zEd4zEl3zUp2zkt1z0x00E1z0U5y0k9x01Fx1FJw1VNv1VRu1lVt11Zs2Fdr2Vhq2lpq2ltp21xo3F1n3V5m3l9l3mFk32Jj4GNj4WRi4mVh4mZg42hf5Gle5Wpd5Wtd5mxc525b529a6HBZ6XFY6XJX6nRX63VW63ZV7HdU7XlT7XpS7ntR73xR735Q8H9P8IBO8YFN8YNM8oRL84VL84dK9IhJ9IlI9YtH9YxG9o1F9o9E95BE95FD95NC+JRB+JVA+Zc/+Zg++Zo++ps9+pw8+p47+586+6E5+6I4/KM4/KU3/KY2/Kg1/Kk0/asz/awz/a4y/a8x/bEw/bIv/bQv/bUu/rct/rgs/ros/rsr/r0q/r4q/sAp/cIp/cMo/cUn/cYn/cgn/com/csm/M0l/M4l/NAl/NIl+9Mk+9Uk+9ck+tgk+tok+dwk+d0l+N8l+OEl9+Il9+Ql9uYm9ugm9ekm9esn9O0n8+4n8/An8vIn8fQm8fUl8Pck8Pkh'
const INFERNO_LUT = 'AAAEAQAFAQEGAQEIAgEKAgIMAgIOAwIQBAMSBAMUBQQXBgQZBwUbCAUdCQYfCgciCwckDAgmDQgpDgkrEAktEQowEgoyFAs0FQs3Fgs5GAw8GQw+GwxBHAxDHgxFHwxIIQxKIwxMJAxPJgxRKAtTKQtVKwtXLQtZLwpbMQpcMgpeNApfNglhOAliOQljOwlkPQllPglmQApnQgpoRApoRQppRwtqSQtqSgxrTAxrTQ1sTw1sUQ5sUg5tVA9tVQ9tVxBuWRBuWhFuXBJuXRJuXxNuYRNuYhRuZBVuZRVuZxZuaRZuahdubBhubRhubxlucRluchpudBpudRtudxxteBxteh1tfB1tfR5tfx5sgB9sgiBshCBrhSFrhyFriCJqiiJqjCNpjSNpjyRpkCVokiVokyZnlSZnlydmmCdmmihlmylknSlknypjoCpjoitioyxhpSxgpi1gqC5fqS5eqy9erTBdrjBcsDFbsTJaszJatDNZtjRYtzVXuTVWujZVvDdUvThTvzlSwDpRwTpQwztPxDxOxj1Nxz5MyD9LykBKy0FJzEJIzkNHz0RG0EVF0kZE00dD1EhC1UpB10s/2Ew+2U092k4821A73VE63lI431M34FU24VY14lc041kz5Fox5Vww5l0v514u6GAt6WEr6mMq62Qp62Yo7Gcm7Wkl7mok72wj724h8G8g8XEf8XMd8nQc83Yb83gZ9HkY9XsX9X0V9n4U9oAT94IS94QQ+IUP+IcO+IkM+YsL+YwK+Y4J+pAI+pIH+pQH+5YG+5cG+5kG+5sG+50H/J8H/KEI/KMJ/KUK/KYM/KgN/KoP/KwR/K4S/LAU/LIW/LQY+7Ya+7gd+7of+7wh+74j+sAm+sIo+sQq+sYt+ccv+cky+cs1+M03+M8699E999NA9tVD9tdG9dlJ9dtM9N1P9N9T9OFW8+Na8+Vd8uZh8uhl8upp8ext8e1x8e918fF58vJ98vSC8/WG8/aK9PiO9fmS9vqW+Pua+fyd+v2h/P+k'
const MAGMA_LUT = 'AAAEAQAFAQEGAQEIAgEJAgILAgINAwMPAwMSBAQUBQQWBgUYBgUaBwYcCAceCQcgCggiCwkkDAkmDQopDgsrEAstEQwvEg0xEw00FA42FQ44Fg87GA89GRA/GhBCHBBEHRFHHhFJIBFLIRFOIhFQJBJTJRJVJxJYKRFaKhFcLBFfLRFhLxFjMRFlMxBnNBBpNhBrOBBsOQ9uOw9wPQ9xPw9yQA90Qg91RA92RRB3RxB4SRB4ShB5TBF6ThF7TxJ7URJ8UhN8VBN9VhR9VxV+WRV+WhZ+XBZ/XRd/Xxh/YBiAYhmAZBqAZRqAZxuAaByBahyBax2BbR2Bbh6BcB+Bch+BcyCBdSGBdiGBeCKBeSKCeyOCfCOCfiSCgCWCgSWBgyaBhCaBhieBiCeBiSiBiymBjCmBjiqBkCqBkSuBkyuAlCyAliyAmC2AmS2Amy5/nC5/ni9/oC9/oTB+ozB+pTF+pjF9qDJ9qjN9qzN8rTR8rjR7sDV7sjV7szZ6tTZ6tzd5uDd5ujh4vDl4vTl3vzp3wDp2wjt1xDx1xTx0xz1zyD5zyj5yzD9xzUBxz0Bw0EFv0kJv00Nu1URt1kVs2EVs2UZr20dq3Ehp3klo30po4Exn4k1m405l5E9k5VBk51Jj6FNi6VRi6lZh61dg7Fhg7Vpf7lte711e8F9e8WBd8mJd8mRc82Vc9Gdc9Glc9Wtc9mxc9m5c93Bc93Jc+HRc+HZc+Xhd+Xld+Xtd+n1e+n9e+oFf+4Nf+4Vg+4dh/Ilh/Ipi/Ixj/I5k/JBl/ZJm/ZRn/ZZo/Zhp/Zpq/Ztr/p1s/p9t/qFu/qNv/qVx/qdy/qlz/qp0/qx2/q53/rB4/rJ6/rR7/rZ8/rd+/rl//ruB/r2C/r+E/sGF/sKH/sSI/saK/siM/sqN/syP/s2Q/s+S/tGU/tOV/tWX/teZ/tia/dqc/dye/d6g/eCh/eKj/eOl/eWn/eep/emq/eus/Oyu/O6w/PCy/PK0/PS2/Pa4/Pe5/Pm7/Pu9/P2/'
const TURBO_LUT = 'MBI7MhVDMxhKNBtRNR5YNiFfNyRmOCdtOSpzOi15Oy+APDKGPTWLPjiRPzuXPz6cQECiQUOnQUasQkmxQku1Q066RFG/RFTDRFbHRVnLRVzPRV7TRmHWRmTaRmbdRmngRmvjR27mR3HpR3PrR3buR3jwR3vyRn30RoD2RoL4RoX6Rof7RYr8RYz9RI/+Q5H+QpT/QZb/QJn/Ppv+PZ7+O6D9OqP8OKX7N6j6Nav4M633Ma/1L7L0LrTyLLfwKrnuKLzrJ77pJcDnI8PkIsXiIMffH8ndHsvaHM3YG9DVGtLSGtTQGdXNGNfKGNnIGNvFGN3CGN7AGOC9GeK7GeO5GuS2HOa0HeeyH+mvIOqsIuuqJeynJ+6kKu+hLPCeL/GbMvKYNfOUOPSRPPWOP/aKQ/eHRviESviATvl9Uvp6Vfp2WftzXfxvYfxsZf1paf1mbf5icf5fdf5cef5Zff9WgP9ThP9RiP9Oi/9Lj/9Jkv9Hlv5Emf5CnP5An/0/of09pPw8p/w6qfs5rPs4r/o3sfk2tPg2t/c1ufY1vPU0vvQ0wfM0w/E0xvA0yO80y+00zew00Oo00uk11Oc11+U12eQ22+I23eA339834d0349s45dk459c56dU569M57NE67s8678068cs68sk69Mc69cU69sM698E6+L45+bw5+ro5+7g4+7Y3/LM2/LE2/a41/aw0/qkz/qcy/qQx/qEw/p4v/pst/pks/pYr/pMq/pAp/Y0n/Yom/Icl/IQj+4Ei+34h+nsf+Xge+XUd+HIc928a9mwZ9WkY9GYX82MV8mAU8V0T8FsS71gR7VUQ7FMP61AO6k4N6EsM50kM5UcL5EUK4kMK4UEJ3z8I3T0I3DsH2jkH2DcG1jUG1DMF0jEF0C8Fzi0EzCsEyioEyCgDxSYDwyUDwSMCviECvCACuR4Ctx0CtBsBshoBrxgBrBcBqRYBpxQBpBMBoRIBnhABmw8BmA4BlQ0BkgsBjgoBiwkCiAgChQcCgQYCfgUCegQD'
const HOT_LUT = 'AAAAAgAABQAACAAACgAADAAADwAAEgAAFAAAFgAAGQAAGwAAHgAAIAAAIwAAJgAAKAAAKgAALQAAMAAAMgAANAAANwAAOQAAPAAAPgAAQQAARAAARgAASAAASwAATgAAUAAAUgAAVQAAWAAAWgAAXAAAXwAAYgAAZAAAZgAAaQAAbAAAbgAAcAAAcwAAdQAAeAAAegAAfQAAgAAAggAAhAAAhwAAigAAjAAAjgAAkQAAlAAAlgAAmAAAmwAAngAAoAAAogAApQAAqAAAqgAArAAArwAAsgAAtAAAtgAAuQAAvAAAvgAAwAAAwwAAxgAAyAAAygAAzQAA0AAA0gAA1AAA1wAA2gAA3AAA3wAA4QAA5AAA5gAA6AAA6wAA7gAA8AAA8wAA9QAA+AAA+gAA/AAA/QIA/gQA/gYA/wgA/woA/w0A/w8A/xIA/xQA/xYA/xkA/xwA/x4A/yAA/yMA/yYA/ygA/yoA/y0A/zAA/zIA/zQA/zcA/zoA/zwA/z4A/0EA/0QA/0YA/0gA/0sA/04A/1AA/1IA/1UA/1gA/1oA/1wA/18A/2IA/2QA/2YA/2kA/2wA/24A/3AA/3MA/3YA/3gA/3oA/30A/4AA/4IA/4QA/4cA/4oA/4wA/44A/5EA/5QA/5YA/5gA/5sA/54A/6AA/6IA/6UA/6gA/6oA/6wA/68A/7IA/7QA/7YA/7kA/7wA/74A/8AA/8MA/8YA/8gA/8oA/80A/9AA/9IA/9QA/9cA/9oA/9wA/94A/+EA/+QA/+YA/+gA/+sA/+4A//AA//IA//UA//gA//oA//wC//0F//4I//8L//8P//8U//8Z//8e//8j//8o//8t//8y//83//88//9B//9G//9L//9Q//9V//9a//9f//9k//9p//9u//9z//94//99//+C//+H//+M//+R//+W//+b//+g//+l//+q//+v//+0//+5//++///D///I///N///S///X///c///h///m///r///w///1///6////'
const COOL_LUT = 'AP//Af7/Av3/A/z/BPv/Bfr/Bvn/B/j/CPf/Cfb/CvX/C/T/DPP/DfL/DvH/D/D/EO//Ee7/Eu3/E+z/FOv/Fer/Fun/F+j/GOf/Geb/GuX/G+T/HOP/HeL/HuH/H+D/IN//Id7/It3/I9z/JNv/Jdr/Jtn/J9j/KNf/Kdb/KtX/K9T/LNP/LdL/LtH/L9D/MM//Mc7/Ms3/M8z/NMv/Ncr/Nsn/N8j/OMf/Ocb/OsX/O8T/PMP/PcL/PsH/P8D/QL//Qb7/Qr3/Q7z/RLv/Rbr/Rrn/R7j/SLf/Sbb/SrX/S7T/TLP/TbL/TrH/T7D/UK//Ua7/Uq3/U6z/VKv/Var/Vqn/V6j/WKf/Wab/WqX/W6T/XKP/XaL/XqH/X6D/YJ//YZ7/Yp3/Y5z/ZJv/ZZr/Zpn/Z5j/aJf/aZb/apX/a5T/bJP/bZL/bpH/b5D/cI//cY7/co3/c4z/dIv/dYr/don/d4j/eIf/eYb/eoX/e4T/fIP/fYL/foH/f4D/gH//gX7/gn3/g3z/hHv/hXr/hnn/h3j/iHf/iXb/inX/i3T/jHP/jXL/jnH/j3D/kG//kW7/km3/k2z/lGv/lWr/lmn/l2j/mGf/mWb/mmX/m2T/nGP/nWL/nmH/n2D/oF//oV7/ol3/o1z/pFv/pVr/pln/p1j/qFf/qVb/qlX/q1T/rFP/rVL/rlH/r1D/sE//sU7/sk3/s0z/tEv/tUr/tkn/t0j/uEf/uUb/ukX/u0T/vEP/vUL/vkH/v0D/wD//wT7/wj3/wzz/xDv/xTr/xjn/xzj/yDf/yTb/yjX/yzT/zDP/zTL/zjH/zzD/0C//0S7/0i3/0yz/1Cv/1Sr/1in/1yj/2Cf/2Sb/2iX/2yT/3CP/3SL/3iH/3yD/4B//4R7/4h3/4xz/5Bv/5Rr/5hn/5xj/6Bf/6Rb/6hX/6xT/7BP/7RL/7hH/7xD/8A//8Q7/8g3/8wz/9Av/9Qr/9gn/9wj/+Af/+Qb/+gX/+wT//AP//QL//gH//wD/'
const OCEAN_LUT = 'AAAAAAABAAACAAADAAAEAAAFAAAGAAAHAAAIAAAJAAAKAAALAAAMAAANAAAOAAAPAAAQAAARAAASAAATAAAUAAAVAAAWAAAXAAAYAAAZAAAaAAAbAAAcAAAdAAAeAAAfAAAgAAAhAAAiAAAjAAAkAAAlAAAmAAAnAAAoAAApAAAqAAArAAAsAAAtAAAuAAAvAAAwAAAxAAAyAAAzAAA0AAA1AAA2AAA3AAA4AAA5AAA6AAA7AAA8AAA9AAA+AAA/AABAAABBAABCAABDAABEAABFAABGAABHAABIAABJAABKAABLAABMAABNAABOAABPAABQAABRAABSAABTAABUAABVAAJWAANXAARYAAZZAAhaAAlbAApcAAxdAA1eAA9fABFgABJhABRiABVjABZkABhlABpmABtnABxoAB5pACBqACFrACJsACRtACZuACdvAChwACpxACxyAC1zAC50ADB1ADJ2ADN3ADR4ADZ5ADh6ADl7ADp8ADx9AD5+AD9/AECAAEKBAESCAEWDAEaEAEiFAEqGAEuHAEyIAE6JAFCKAFGLAFKMAFSNAFaOAFePAFiQAFqRAFySAF2TAF6UAGCVAGKWAGOXAGSYAGaZAGiaAGmbAGqcAGydAG6eAG+fAHCgAHKhAHSiAHWjAHakAHilAHqmAHunAHyoAH6pAICqA4GrBoKsCYStDIauD4evEoiwFYqxGIyyG42zHo60IZC1JJK2J5O3KpS4LZa5MJi6M5m7Npq8OZy9PJ6+P5+/QqHARaLBSKTCS6XDTqbEUajFVKrGV6vHWqzIXa7JYLDKY7HLZrLMabTNbLbOb7fPcrjQdbrReLzSe73Tfr7UgcDVhMLWh8PXisTYjcbZkMjak8nblsrcmczdnM7en8/fotDgpdLhqNTiq9XjrtbksdjltNrmt9vnutzovd7pwODqw+HrxuLsyeTtzObuz+fv0ujw1erx2Ozy2+3z3u704fD15PL25/P36vT47fb58Pj68/n79vv8+fz9/P7+////'
const RAINBOW_LUT = '/wAA/wIA/wUA/wgA/woA/wwA/w8A/xIA/xQA/xYA/xkA/xsA/x4A/yAA/yMA/yYA/ygA/yoA/y0A/zAA/zIA/zQA/zcA/zkA/zwA/z4A/0EA/0QA/0YA/0gA/0sA/04A/1AA/1IA/1UA/1gA/1oA/1wA/18A/2IA/2QA/2YA/2kA/2wA/24A/3AA/3MA/3UA/3gA/3oA/30A/4AA/4IA/4QA/4cA/4oA/4wA/44A/5EA/5QA/5YA/5gA/5sA/54A/6AA/6IA/6UA/6gA/6oA/6wA/68A/7IA/7QA/7YA/7kA/7wA/74A/8AA/8MA/8YA/8gA/8oA/80A/9AA/9IA/9QA/9cA/9oA/9wA/98A/+EA/+QA/+YA/+gA/+sA/+4A//AA//MA//UA//gA//oA//wA/P0A+P4A9P4A8P8A6/8A5v8A4f8A3P8A1/8A0v8Azf8AyP8Aw/8Avv8Auf8AtP8Ar/8Aqv8Apf8AoP8Am/8Alv8Akf8AjP8Ah/8Agv8Aff8AeP8Ac/8Abv8Aaf8AZP8AX/8AWv8AVf8AUP8AS/8ARv8AQf8APP8AN/8AMv8ALf8AKP8AI/8AHv8AGf8AFP8AD/8AC/4BB/0CA/wDAPoFAPUKAPAPAOsUAOYZAOEeANwjANcoANItAM0yAMg3AMM8AL5BALlGALRLAK9QAKpVAKVaAKBfAJtkAJZpAJFuAIxzAId4AIJ9AH2CAHiHAHOMAG6RAGmWAGSbAF+gAFqlAFWqAFCvAEu0AEa5AEG+ADzDADfIADLNAC3SACjXACPcAB7hABnmABTrAA/wAAr1AQf4AwX6BQP8BwH+CgD/DQD/EQD/FAD/FwD/GwD/HgD/IQD/JQD/KAD/KwD/LwD/MgD/NQD/OQD/PAD/PwD/QwD/RgD/SQD/TQD/UAD/UwD/VwD/WgD/XQD/YQD/ZAD/ZwD/awD/bgD/cQD/dQD/eAD/ewD/fwD/ggD/hQD/iQD/jAD/jwD/kwD/lgD/mQD/nQD/oAD/owD/pwD/qgD/'
const PARULA_LUT = 'NSqHMyyKMi6NMC+PLzGSLTOVLDWYKjebKDieJzqgJTyjJD6mIj+pIUGsH0OvHUWxHEe0Gki3GUq6F0y9Fk6/FFDCElHFEVPID1XLDlfODFjQC1rTCVzWCF7ZBmDcBGHfA2PhBGThBGXgBWbgBWffBmjfBmrfB2veB2zeCG3dCG7dCW/cCXDcCnHcC3LbC3PbDHTaDHXaDXbaDXfZDnjZDnrYD3vYD3zYEH3XEH7XEX/WEYDWEoHWE4LVE4PVFITUFIXUE4bTE4fTE4jTEorSEovSEYzREY3REI7QEI/QD5DQD5HPD5LPDpPODpTODZXNDZbNDJfMDJjMDJrMC5vLC5zLCp3KCp7KCZ/JCaDJCKHICKLICKPIB6THB6XHBqbGB6fGCKjECqjDC6nCDanBDqq/EKu+Equ9E6y7Fay6Fq25GK24Ga62G6+1Ha+0HrCzILCxIbGwI7GvJLKuJrOsKLOrKbSqK7SpLLWnLrWmL7alMbakM7eiNLihNrigN7mfObmdPLmcP7mbQrqZRbqYSLqXSrqVTbqUULuTU7uRVruQWLuOW7uNXryMYbyKZLyJZ7yIabyGbLyFb72Ecr2Cdb2BeL2Aer1+fb59gL58g756hr55iL54i792jr91kb90k79ylr9ymL9xmr5wnL5vn75uob5to75spb5rp71qqr1prL1orr1osL1ns71mtb1lt7xkubxju7xivrxhwLxgwrxfxLtex7teybtdy7tczbtb0Lta0rtZ1LpY1rpX2LpW2rpV27tU3LxT3bxR3r1Q375P4L5O4r9M479L5MBK5cFJ5sFH58JG6MNF6cND6sRC68VB7MVA7sY+78Y98Mc88cg78sg588k49Mo39co29ss098sz+cwy+s0x+80v/M4u/M8t/NEs/NIr/NMq/NUp+9Yo+9gn+9km+9sl+9wk+90j+98i++Ah++Ig++Mf++Qe+uYd+ucc+ukb+uoa+usZ+u0Y+u4X+vAW+vEV+vMU+fQT+fUS+fcR+fgQ+foP+fsO'
const CIVIDIS_LUT = 'ACJOACNPACRRACVTACVUACZWACdYAChZAChbACldACpfACphACtiACxkACxmAC1oAC5qAC5sAC9tADBvADBwADFwADFxATJxBTNxCDNwDDRwDzVwEjVwFDZwFjdwGDdvGjhvHDlvHjpvIDpvITtuIzxuJDxuJj1uJz5uKT9uKj9tK0BtLUFtLkFtL0JtMUNtMkNtM0RtNEVsNUVsNkZsOEdsOUhsOkhsO0lsPEpsPUpsPktsP0xsQExsQU1sQk5sQ05sRE9sRVBsRlFsR1FsSFJsSVNsSlNsS1RsTFVsTVVsTlZsT1dsUFdsUVhtUlltU1ptVFptVVttVVxtVlxtV11tWF5tWV5uWl9uW2BuXGFuXWFuXmJuXmNvX2NvYGRvYWVvYmVvY2ZwZGdwZWhwZWhwZmlwZ2pxaGpxaWtxamxxa21ybG1ybG5ybW9ybm9zb3BzcHFzcXJ0cnJ0cnN0c3R1dHR1dXV1dnZ2d3d2d3d3eHh3eXl3enp4e3p4fHt4fXx4fnx4fn14f354gH94gX94goB5g4F5hIJ5hYJ5hoN5h4R4iIV4iYV4ioZ4i4d4jIh4jYh4jol4j4p4kIt4kYt4kox4ko14k454lI53lY93lpB3l5F3mJJ3mZJ3mpN2m5R2nJV2nZV2npZ2n5d1oJh1oZl1opl1o5p0pJt0pZx0ppx0p51zqJ5zqZ9zqqBzq6ByrKFyraJyrqNxr6RxsKVxsaVws6ZwtKdvtahvtqlvt6luuKpuuattuqxtu61tvK5sva5svq9rv7BrwLFqwbJqwrNpw7NpxLRoxbVoxrZnx7dnyLhmyblly7llzLpkzbtjzrxjz71i0L5i0b9h0sBg08Bf1MFf1cJe1sNd18Rc2cVc2sZb28da3MhZ3chY3slY38pX4MtW4cxV4s1U5M5T5c9S5tBR59FQ6NJP6dNO6tNM69RL7dVK7tZJ79dI8NhG8dlF8tpE89tC9dxB9t0/994++N88+eA6++E4/OI2/eM0/uQ0/uU1/uY2/ug4'

export const jetColor = lutColor(JET_LUT)
export const viridisColor = lutColor(VIRIDIS_LUT)
export const plasmaColor = lutColor(PLASMA_LUT)
export const infernoColor = lutColor(INFERNO_LUT)
export const magmaColor = lutColor(MAGMA_LUT)
export const turboColor = lutColor(TURBO_LUT)
export const hotColor = lutColor(HOT_LUT)
export const coolColor = lutColor(COOL_LUT)
export const oceanColor = lutColor(OCEAN_LUT)
export const rainbowColor = lutColor(RAINBOW_LUT)
export const parulaColor = lutColor(PARULA_LUT)
export const cividisColor = lutColor(CIVIDIS_LUT)

export const COLORMAPS: Record<string, (v: number) => [number, number, number]> = {
  Viridis: viridisColor,
  Plasma: plasmaColor,
  Inferno: infernoColor,
  Magma: magmaColor,
  Turbo: turboColor,
  Jet: jetColor,
  Hot: hotColor,
  Cool: coolColor,
  Parula: parulaColor,
  Cividis: cividisColor,
  Rainbow: rainbowColor,
  Ocean: oceanColor,
}

/** Applies a colormap function to a single-channel byte Mat, producing a new BGR Mat. */
export function applyColormap(cv: any, gray: any, colorFn: (v: number) => [number, number, number]): any {
  const lut: [number, number, number][] = []
  for (let i = 0; i < 256; i++) lut[i] = colorFn(i)

  const out = new cv.Mat(gray.rows, gray.cols, cv.CV_8UC3)
  const src = gray.data as Uint8Array
  const dst = out.data as Uint8Array
  for (let i = 0, px = 0; i < src.length; i++, px += 3) {
    const [r, g, b] = lut[src[i]]
    dst[px] = b
    dst[px + 1] = g
    dst[px + 2] = r
  }
  return out
}
