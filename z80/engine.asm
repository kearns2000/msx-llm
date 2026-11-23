; =============================================================================
;  msx-llm engine -- a ternary-weight RNN language model for a stock MSX1
;  Z80 @ 3.58 MHz, 64 KB RAM, ASCII8 MegaROM.  Assemble with sjasmplus:
;     sjasmplus --sym=build/msxllm.sym z80/engine.asm
;  Needs build/model_data.bin (export/pack.py) -> writes build/msxllm.rom
;
;  The Python twin of every arithmetic step is train/intref.py.
; =============================================================================
        OUTPUT "build/msxllm.rom"

; ---- BIOS / system variables ------------------------------------------------
CHGET   equ 009Fh
CHPUT   equ 00A2h
INITXT  equ 006Ch
CSRX    equ 0F3DDh
LINL40  equ 0F3AEh
CLIKSW  equ 0F3DBh
JIFFY   equ 0FC9Eh
EXPTBL  equ 0FCC1h
SLTTBL  equ 0FCC5h              ; BIOS copy of each expanded slot's subslot reg
CHSNS   equ 009Ch
INIGRP  equ 0072h
LDIRVM  equ 005Ch
FILVRM  equ 0056h
CGTABL  equ 0004h
FORCLR  equ 0F3E9h
BAKCLR  equ 0F3EAh
BDRCLR  equ 0F3EBh

; ---- cartridge --------------------------------------------------------------
BANKREG equ 6800h               ; ASCII8: bank for 6000h-7FFFh window
WIN     equ 6000h               ; data window (code lives in bank 0 @ 4000h)

; ---- RAM map (page 2 is switched to RAM at boot) -----------------------------
TBL     equ 8000h               ; 16 groups x 2 pages: 16-bit partial sums
TBLH    equ TBL>>8
Y       equ 0A000h              ; 1024 x s16 accumulators / logits
VEC     equ 0A800h              ; [x (E=128) ; h (H)] int8, <= 512 bytes
HVEC    equ VEC+128
LUTZ    equ 0AA00h              ; sigmoid LUT (Q7)
LUTN    equ 0AB00h              ; tanh LUT (s8)
LUTE    equ 0AC00h              ; softmax weight LUT (u8)
PROMPT  equ 0AD00h              ; ' ' + typed text
HIST    equ 0AE00h              ; 256 x u16 token history
ATAB    equ 0B100h              ; 9 x s16 partial sums of inputs 0,1
BTAB    equ 0B120h              ; 9 x s16 partial sums of inputs 2,3
STACK   equ 0C000h               ; grows down into B300h-BFFFh (page 2 only)

; ---- manifest (copied from bank 1 offset 0 to MAN) ---------------------------
M_E     equ 8
M_H     equ 10
M_V     equ 12
M_NCG   equ 14
M_NCO   equ 15
M_KZ    equ 16
M_KN    equ 17
M_KE    equ 18
M_TOPK  equ 19
M_RCZ   equ 20
M_RCN   equ 22
M_MAXNEW equ 24
M_BLK   equ 32
B_VOCAB equ 0
B_EMB   equ 1
B_BG    equ 2
B_WG    equ 3
B_BO    equ 4
B_WO    equ 5
B_LUTZ  equ 6
B_LUTN  equ 7
B_LUTE  equ 8

; =============================================================================
        ORG 4000h
        db "AB"
        dw start
        dw 0,0,0,0,0,0

start:
        di                      ; still on the BIOS stack (page 3) here: our
        call ram_page2          ; stack lives in page 2, so it must be RAM first
        ld hl,8000h             ; need RAM at 8000h-BFFFh (>= 32 KB machine)
        ld a,(hl)
        cpl
        ld (hl),a
        cp (hl)
        jp nz,no_ram
        cpl
        ld (hl),a
        ld sp,STACK
        ld a,1                  ; manifest: bank 1, offset 0
        ld hl,0
        ld de,MAN
        ld bc,96
        call romcopy
        ld a,B_LUTZ
        ld de,LUTZ
        call load256
        ld a,B_LUTN
        ld de,LUTN
        call load256
        ld a,B_LUTE
        ld de,LUTE
        call load256
        call splash
        ld a,40
        ld (LINL40),a
        call INITXT
        xor a
        ld (CLIKSW),a
        ld (greedy),a
        ld hl,msg_banner
        call puts

main_loop:
        ld hl,msg_prompt
        call puts
        call read_line
        di
        ld hl,(JIFFY)           ; seed = frame counter at <RETURN>
        ld a,h
        or l
        jr nz,.seed
        inc hl
.seed:  ld (rng),hl
        ld (seed0),hl
        call encode             ; HIST = <eos> + prompt tokens
        ld hl,HVEC              ; h = 0
        ld de,HVEC+1
        ld bc,(MAN+M_H)
        dec bc
        ld (hl),0
        ldir
        ld hl,0
        ld (fidx),hl
.feed:  ld hl,(fidx)
        add hl,hl
        ld de,HIST
        add hl,de
        ld e,(hl)
        inc hl
        ld d,(hl)
        ex de,hl
        call feed
        ld hl,(fidx)
        inc hl
        ld (fidx),hl
        ld de,(ntok)
        or a
        sbc hl,de
        jr c,.feed
        xor a
        ld (ngen),a
gen_loop:
        call out_layer
        call sample             ; HL = token
        ld (last_tok),hl
        ex de,hl
        ld hl,(ntok)
        push hl
        add hl,hl
        ld bc,HIST
        add hl,bc
        ld (hl),e
        inc hl
        ld (hl),d
        pop hl
        inc hl
        ld (ntok),hl
ev_token:                       ; test hook: a token was just sampled
        ld a,d
        or e
        jr z,ev_done            ; <eos>
        ex de,hl
        push hl
        call print_tok
        pop hl
        ld a,(ngen)
        inc a
        ld (ngen),a
        ld b,a
        ld a,(MAN+M_MAXNEW)
        cp b
        jr z,ev_done
        call feed
        jr gen_loop
ev_done:                        ; test hook: generation finished
        ld a,13
        call CHPUT
        ld a,10
        call CHPUT
        jp main_loop

; -----------------------------------------------------------------------------
; feed: HL = token.  x = EMB[token]; y = Wg.[x;h] + bg; h = update(h, y)
; -----------------------------------------------------------------------------
feed:
        push hl
        ld a,B_EMB
        call getblk             ; A = bank, HL = offset (row aligned)
        ld (tmp_off),hl
        ld c,a
        pop hl
        push hl
        add hl,hl
        add hl,hl               ; H = token >> 6  (64 rows of 128 per bank)
        ld a,h
        add a,c
        ld c,a
        pop hl
        ld a,l
        and 63
        ld h,a
        ld l,0
        srl h
        rr l                    ; (token & 63) * 128
        ld de,(tmp_off)         ; EMB starts a bank (export/pack.py asserts it),
        add hl,de               ; so this offset is 0 and rows never straddle
        ld a,c
        ld de,VEC
        ld bc,128
        call romcopy
        ld hl,(MAN+M_H)
        add hl,hl
        ld (mv_rows),hl
        ld a,(MAN+M_NCG)
        ld (mv_nch),a
        ld hl,VEC
        ld (mv_in),hl
        ld a,B_WG
        ld b,B_BG
        call matvec
        ; fall through
; -----------------------------------------------------------------------------
; gate_update: z = LUTZ[idx(Y[j])], n = LUTN[idx(Y[H+j])],
;              h += (z*(n-h)+64) >> 7      for j = 0..H-1
; -----------------------------------------------------------------------------
gate_update:
        ld hl,Y
        ld (gu_z),hl
        ld hl,(MAN+M_H)
        ld (gu_cnt),hl
        add hl,hl
        ld de,Y
        add hl,de
        ld (gu_n),hl
        ld hl,HVEC
        ld (gu_h),hl
.l:     ld hl,(gu_z)
        ld e,(hl)
        inc hl
        ld d,(hl)
        inc hl
        ld (gu_z),hl
        ex de,hl
        ld de,(MAN+M_RCZ)
        ld a,(MAN+M_KZ)
        call lutidx
        xor 80h
        ld l,a
        ld h,LUTZ>>8
        ld a,(hl)
        ld (gu_zv),a
        ld hl,(gu_n)
        ld e,(hl)
        inc hl
        ld d,(hl)
        inc hl
        ld (gu_n),hl
        ex de,hl
        ld de,(MAN+M_RCN)
        ld a,(MAN+M_KN)
        call lutidx
        xor 80h
        ld l,a
        ld h,LUTN>>8
        ld a,(hl)               ; n (s8)
        ld e,a
        rla
        sbc a,a
        ld d,a                  ; DE = n
        ld hl,(gu_h)
        ld a,(hl)
        ld c,a                  ; C = h (kept for the final add)
        rla
        sbc a,a
        ld b,a                  ; BC = h
        ex de,hl
        or a
        sbc hl,bc
        ex de,hl                ; DE = n - h
        ld a,(gu_zv)
        call mulzd              ; HL = z * (n-h)
        ld de,64
        add hl,de
        add hl,hl               ; H = ((p+64) >> 7) & 255
        ld a,h
        add a,c
        ld hl,(gu_h)
        ld (hl),a
        inc hl
        ld (gu_h),hl
        ld hl,(gu_cnt)
        dec hl
        ld (gu_cnt),hl
        ld a,h
        or l
        jr nz,.l
        ret

; lutidx: HL = y, DE = rounding const, A = shift k  ->  A = clamp((y+rc)>>k)
lutidx:
        add hl,de
        or a
        jr z,.ns
        ld b,a
.sh:    sra h
        rr l
        djnz .sh
.ns:    ld a,l
        rla                     ; carry = bit 7 of L
        ld a,h
        adc a,0                 ; 0 iff HL fits in int8
        jr nz,.sat
        ld a,l
        ret
.sat:   bit 7,h
        ld a,7Fh
        ret z
        ld a,80h
        ret

; mulzd: A = z (u8), DE = d (s16)  ->  HL = z*d mod 2^16   (no B/C use)
mulzd:
        ld hl,0
        REPT 8
        add hl,hl
        add a,a
        jr nc,$+3
        add hl,de
        ENDR
        ret

; -----------------------------------------------------------------------------
; out_layer: Y[0..V-1] = Wo.h + bo
; -----------------------------------------------------------------------------
out_layer:
        ld hl,(MAN+M_V)
        ld (mv_rows),hl
        ld a,(MAN+M_NCO)
        ld (mv_nch),a
        ld hl,HVEC
        ld (mv_in),hl
        ld a,B_WO
        ld b,B_BO
        ; fall through
; -----------------------------------------------------------------------------
; matvec: A = weight block, B = bias block; mv_rows, mv_nch, mv_in set.
;   Y = bias; for each 64-input chunk: build 16 partial-sum tables, then for
;   every row add 16 table entries (one per weight byte = 4 ternary weights).
; -----------------------------------------------------------------------------
matvec:
        push af
        ld a,b
        call getblk
        ld de,Y
        ld bc,(mv_rows)
        sla c
        rl b
        call romcopy            ; Y = bias
        pop af
        call getblk
        ld (mv_wbank),a
        ld (mv_woff),hl
        ld a,(mv_nch)
        ld (mv_c),a
.chunk: ld hl,(mv_in)
        call build_tables
        ld hl,(mv_in)
        ld de,64
        add hl,de
        ld (mv_in),hl
        ld hl,(mv_rows)
        ld (mv_left),hl
        ld hl,Y
        ld (mv_y),hl
.seg:   ld hl,(mv_woff)         ; normalise (bank, offset)
        ld a,h
        cp 20h
        jr c,.inb
        sub 20h
        ld h,a
        ld (mv_woff),hl
        ld a,(mv_wbank)
        inc a
        ld (mv_wbank),a
.inb:   ex de,hl
        ld hl,2000h
        or a
        sbc hl,de
        ld b,4
.sh:    srl h
        rr l
        djnz .sh                ; HL = rows left in this bank
        ld de,(mv_left)
        push hl
        or a
        sbc hl,de
        pop hl
        jr c,.k1                ; avail < left
        ex de,hl                ; HL = left
.k1:    ld a,h
        or a
        jr z,.k2
        ld hl,256
.k2:    ld (mv_n),hl
        ex de,hl
        ld hl,(mv_left)
        or a
        sbc hl,de
        ld (mv_left),hl
        ld a,(mv_wbank)
        ld (BANKREG),a
        di
        ld a,(mv_n)             ; 256 -> 0, fine for dec c / jp nz
        ld hl,(mv_y)
        ld c,a
        exx                     ; HL' = y pointer, C' = row count
        ld (save_sp),sp
        ld hl,(mv_woff)
        ld de,WIN
        add hl,de
        ld sp,hl                ; SP = weight stream

; ---- the inner loop: one row of one 64-input chunk ---------------------------
        MACRO ACC               ; DE += T[g][L]; advance to next group's page
        ld a,(hl)
        add a,e
        ld e,a
        inc h
        ld a,(hl)
        adc a,d
        ld d,a
        inc h
        ENDM
.krow:   ld h,TBLH
        pop bc                  ; 2 weight bytes = 8 ternary weights
        ld l,c
        ld e,(hl)
        inc h
        ld d,(hl)
        inc h
        ld l,b
        ACC
        REPT 7
        pop bc
        ld l,c
        ACC
        ld l,b
        ACC
        ENDR
        ld a,e                  ; *y++ += DE  (y pointer lives in HL')
        exx
        add a,(hl)
        ld (hl),a
        inc hl
        exx
        ld a,d
        exx
        adc a,(hl)
        ld (hl),a
        inc hl
        dec c
        exx
        jp nz,.krow
; -----------------------------------------------------------------------------
        ld (mv_sp),sp
        ld sp,(save_sp)
        exx
        ld (mv_y),hl
        exx
        ld hl,(mv_sp)
        ld de,-WIN
        add hl,de
        ld (mv_woff),hl
        ld hl,(mv_left)
        ld a,h
        or l
        jp nz,.seg
        ld a,(mv_c)
        dec a
        ld (mv_c),a
        jp nz,.chunk
        ret

; -----------------------------------------------------------------------------
; build_tables: HL -> 64 int8 inputs.  For group g (inputs 4g..4g+3) fill
;   T[g][code] (lo byte page TBLH+2g, hi byte page +1) for the 81 valid codes
;   code = c0 | c1<<2 | c2<<4 | c3<<6, c = 00:0 01:+1 11:-1
; -----------------------------------------------------------------------------
build_tables:
        ld (bt_src),hl
        ld a,TBLH
        ld (bt_page),a
        ld a,16
        ld (bt_g),a
.g:     ld hl,(bt_src)
        ld ix,ATAB
        call mk_pair
        ld hl,(bt_src)
        inc hl
        inc hl
        ld ix,BTAB
        call mk_pair
        ld hl,(bt_src)
        ld de,4
        add hl,de
        ld (bt_src),hl
        ld a,(bt_page)
        ld h,a
        call fill
        ld a,(bt_page)
        add a,2
        ld (bt_page),a
        ld a,(bt_g)
        dec a
        ld (bt_g),a
        jr nz,.g
        ret

; mk_pair: HL -> (a, b) int8.  IX[0..8] = b*cb + a*ca for the 9 nibble codes
;   in order 0,1,3,4,5,7,12,13,15  (cb, ca in {0,+1,-1})
        MACRO TRIP ofs          ; IX[ofs..] = base, base+a, base-a  (HL=base)
        ld (ix+ofs+0),l
        ld (ix+ofs+1),h
        push hl
        add hl,bc
        ld (ix+ofs+2),l
        ld (ix+ofs+3),h
        pop hl
        or a
        sbc hl,bc
        ld (ix+ofs+4),l
        ld (ix+ofs+5),h
        ENDM
mk_pair:
        ld a,(hl)
        ld c,a
        rla
        sbc a,a
        ld b,a                  ; BC = a
        inc hl
        ld a,(hl)
        ld e,a
        rla
        sbc a,a
        ld d,a                  ; DE = b
        ld hl,0
        TRIP 0
        ld h,d
        ld l,e
        TRIP 6
        ld hl,0
        or a
        sbc hl,de
        TRIP 12
        ret

; fill: H = lo page of this group.  T[hi<<4|lo] = ATAB[lo] + BTAB[hi]
;   fully unrolled over the 81 valid codes (L walks the 9 lo codes per row)
        MACRO ENT               ; T[L] = ATAB[i] + BC ; DE -> ATAB[i]
        ld a,(de)
        add a,c
        ld (hl),a
        inc e
        inc h
        ld a,(de)
        adc a,b
        ld (hl),a
        dec h
        inc e
        ENDM
        MACRO FROW bofs, lbase
        ld bc,(BTAB+bofs)
        ld de,ATAB
        ld l,lbase
        ENT
        inc l
        ENT
        inc l
        inc l
        ENT
        inc l
        ENT
        inc l
        ENT
        inc l
        inc l
        ENT
        ld l,lbase+12
        ENT
        inc l
        ENT
        inc l
        inc l
        ENT
        ENDM
fill:   FROW 0,00h
        FROW 2,10h
        FROW 4,30h
        FROW 6,40h
        FROW 8,50h
        FROW 10,70h
        FROW 12,0C0h
        FROW 14,0D0h
        FROW 16,0F0h
        ret

; -----------------------------------------------------------------------------
; sample: top-K over Y[0..V-1], LUT softmax, xorshift draw.  -> HL = token
;   topv holds values XOR 8000h so unsigned compares order signed values.
; -----------------------------------------------------------------------------
sample:
        ld a,(MAN+M_TOPK)
        ld b,a
        ld hl,TOPV
.i:     xor a
        ld (hl),a               ; topv = -32768 (biased 0000h)
        inc hl
        ld (hl),a
        dec hl
        push hl
        ld de,32
        add hl,de
        ld (hl),a               ; topi = 0
        inc hl
        ld (hl),a
        pop hl
        inc hl
        inc hl
        djnz .i
        ld a,(MAN+M_TOPK)
        dec a
        add a,a
        ld e,a
        ld d,0
        ld hl,TOPV
        add hl,de
        ld (sm_tlast),hl
        ld hl,(MAN+M_V)
        ld (sm_cnt),hl
        ld hl,0
        ld (sm_i),hl
        ld ix,Y
.s:     ld e,(ix+0)
        ld a,(ix+1)
        xor 80h
        ld d,a                  ; DE = biased logit
        ld hl,(sm_tlast)
        ld a,(hl)
        inc hl
        ld h,(hl)
        ld l,a
        or a
        sbc hl,de               ; carry iff topv[K-1] < v
        call c,insert
        inc ix
        inc ix
        ld hl,(sm_i)
        inc hl
        ld (sm_i),hl
        ld hl,(sm_cnt)
        dec hl
        ld (sm_cnt),hl
        ld a,h
        or l
        jr nz,.s
        ld a,(greedy)
        or a
        jr z,.smp
        ld hl,(TOPV+32)
        ret
.smp:   ld a,(MAN+M_TOPK)
        ld b,a
        ld ix,TOPV
        ld iy,WBUF
        ld hl,0
        ld (sm_sum),hl
.wl:    ld hl,(TOPV)
        ld e,(ix+0)
        ld d,(ix+1)
        or a
        sbc hl,de               ; d = topv[0] - topv[j] >= 0
        ld a,(MAN+M_KE)
        or a
        jr z,.nsh
        ld c,a
.sh2:   srl h
        rr l
        dec c
        jr nz,.sh2
.nsh:   ld a,h
        or a
        ld a,l
        jr z,.ok
        ld a,255
.ok:    ld l,a
        ld h,LUTE>>8
        ld a,(hl)
        ld (iy+0),a
        ld e,a
        ld d,0
        ld hl,(sm_sum)
        add hl,de
        ld (sm_sum),hl
        inc iy
        inc ix
        inc ix
        djnz .wl
        call rand
        ld a,h                  ; r >> 8
        ld de,(sm_sum)
        call mul8_16            ; C:H:L = A * DE
        ld l,h
        ld h,c                  ; HL = t = (r>>8)*S >> 8
        ld ix,TOPV
        ld iy,WBUF
        ld de,0
.walk:  ld a,(iy+0)
        add a,e
        ld e,a
        ld a,d
        adc a,0
        ld d,a                  ; cum += w
        push hl
        or a
        sbc hl,de
        pop hl
        jr c,.pick              ; t < cum
        inc iy
        inc ix
        inc ix
        jr .walk
.pick:  ld l,(ix+32)
        ld h,(ix+33)
        ret

; insert: DE = biased value, (sm_i) = index; keeps TOPV sorted descending
insert:
        push ix
        ld ix,(sm_tlast)
        ld a,(MAN+M_TOPK)
        dec a
        ld b,a
.w:     ld a,b
        or a
        jr z,.put
        ld l,(ix-2)
        ld h,(ix-1)
        or a
        sbc hl,de               ; carry iff topv[j-1] < v
        jr nc,.put
        ld a,(ix-2)
        ld (ix+0),a
        ld a,(ix-1)
        ld (ix+1),a
        ld a,(ix+30)
        ld (ix+32),a
        ld a,(ix+31)
        ld (ix+33),a
        dec ix
        dec ix
        dec b
        jr .w
.put:   ld (ix+0),e
        ld (ix+1),d
        ld hl,(sm_i)
        ld (ix+32),l
        ld (ix+33),h
        pop ix
        ret

; mul8_16: A * DE -> C:HL (24-bit)
mul8_16:
        ld hl,0
        ld c,0
        ld b,8
.l:     add hl,hl
        rl c
        add a,a
        jr nc,.n
        add hl,de
        jr nc,.n
        inc c
.n:     djnz .l
        ret

; rand: 16-bit xorshift (7,9,8) after John Metcalf -> HL
rand:   ld hl,(rng)
        ld a,h
        rra
        ld a,l
        rra
        xor h
        ld h,a
        ld a,l
        rra
        ld a,h
        rra
        xor l
        ld l,a
        xor h
        ld h,a
        ld (rng),hl
        ret

; -----------------------------------------------------------------------------
; encode: greedy longest match of PROMPT[0..plen) against the vocab.
;   HIST = <eos>, ids...;  unknown chars are skipped.
; -----------------------------------------------------------------------------
encode:
        ld hl,0
        ld (HIST),hl
        inc hl
        ld (ntok),hl
        ld a,B_VOCAB
        call getblk
        ld (BANKREG),a
        ld de,WIN
        add hl,de
        ld (vocab_ptr),hl
        xor a
        ld (en_pos),a
.next:  ld a,(en_pos)
        ld b,a
        ld a,(plen)
        sub b
        ret z
        ld (en_rem),a
        xor a
        ld (en_blen),a
        ld hl,0
        ld (en_id),hl
        ld (en_bid),hl
        ld hl,(MAN+M_V)
        ld (en_cnt),hl
        ld hl,(vocab_ptr)
.tok:   ld c,(hl)               ; C = token length
        ld a,(en_blen)
        cp c
        jr nc,.skip             ; not longer than the best so far
        ld a,(en_rem)
        cp c
        jr c,.skip              ; longer than what is left
        push hl
        inc hl
        ld a,(en_pos)
        ld e,a
        ld d,PROMPT>>8
        ld b,c
.cmp:   ld a,(de)
        cp (hl)
        jr nz,.ne
        inc hl
        inc de
        djnz .cmp
        ld a,c
        ld (en_blen),a
        ld hl,(en_id)
        ld (en_bid),hl
.ne:    pop hl
.skip:  ld b,0
        add hl,bc
        inc hl
        ld de,(en_id)
        inc de
        ld (en_id),de
        ld de,(en_cnt)
        dec de
        ld (en_cnt),de
        ld a,d
        or e
        jr nz,.tok
        ld a,(en_blen)
        or a
        jr nz,.emit
        ld a,(en_pos)
        inc a
        ld (en_pos),a
        jr .next
.emit:  ld b,a
        ld a,(en_pos)
        add a,b
        ld (en_pos),a
        ld hl,(ntok)
        push hl
        add hl,hl
        ld de,HIST
        add hl,de
        ld de,(en_bid)
        ld (hl),e
        inc hl
        ld (hl),d
        pop hl
        inc hl
        ld (ntok),hl
        jp .next

; print_tok: HL = token id.  Word wrap: a token starting with a space must
;   end by column 39, any other token by column 40 (so punctuation after a
;   word still fits on its line); otherwise it starts a new line, dropping
;   its leading space.  The BIOS would otherwise wrap mid-token at column 40.
print_tok:
        ld (pt_id),hl
        ld a,B_VOCAB
        call getblk
        ld (BANKREG),a
        ld de,WIN
        add hl,de
        ld bc,(pt_id)
.f:     ld a,b
        or c
        jr z,.got
        ld e,(hl)
        ld d,0
        add hl,de
        inc hl
        dec bc
        jr .f
.got:   ld b,(hl)
        inc hl
        ld a,b
        or a
        ret z
        ld c,41                 ; a token may end in column 40 ...
        ld a,(hl)
        cp ' '
        jr nz,.lim
        dec c                   ; ... but a word ends by column 39, keeping
.lim:   ld a,(CSRX)             ; column 40 free for a following , . ! ?
        cp 2
        jr c,.pr                ; already at the start of a line
        dec a
        add a,b
        cp c
        jr c,.pr
        push hl
        push bc
        ld a,13
        call CHPUT
        ld a,10
        call CHPUT
        pop bc
        pop hl
        ld a,(hl)               ; drop a leading space at the new line
        cp ' '
        jr nz,.pr
        inc hl
        dec b
        ret z
.pr:    ld a,(hl)
        push hl
        push bc
        call CHPUT
        pop bc
        pop hl
        inc hl
        djnz .pr
        ret

; read_line: PROMPT = ' ' + typed text (spaces collapsed, trailing stripped)
read_line:
        ld a,' '
        ld (PROMPT),a
        ld a,1
        ld (plen),a
ev_ready:                       ; test hook: waiting for keyboard input
.k:     ei
        call CHGET
        di
        cp 13
        jr z,.done
        cp 8
        jr z,.bs
        cp 32
        jr c,.k
        cp 127
        jr nc,.k
        ld c,a
        ld a,(plen)
        cp 120
        jr nc,.k
        ld a,c
        cp ' '
        jr nz,.add
        ld a,(plen)
        dec a
        ld l,a
        ld h,PROMPT>>8
        ld a,(hl)
        cp ' '
        jr z,.k
.add:   ld a,(plen)
        ld l,a
        ld h,PROMPT>>8
        ld (hl),c
        inc a
        ld (plen),a
        ld a,c
        call CHPUT
        jr .k
.bs:    ld a,(plen)
        cp 2
        jr c,.k
        dec a
        ld (plen),a
        ld a,1Dh
        call CHPUT
        ld a,' '
        call CHPUT
        ld a,1Dh
        call CHPUT
        jr .k
.done:  ld a,(plen)
        cp 2
        jr nc,.strip
        xor a                   ; nothing typed: empty prompt
        ld (plen),a
        ret
.strip:
        dec a
        ld l,a
        ld h,PROMPT>>8
        ld a,(hl)
        cp ' '
        ret nz
        ld a,(plen)
        dec a
        ld (plen),a
        ret

; -----------------------------------------------------------------------------
; utilities
; -----------------------------------------------------------------------------
puts:   ld a,(hl)
        or a
        ret z
        push hl
        call CHPUT
        pop hl
        inc hl
        jr puts

; getblk: A = block index -> A = bank, HL = offset in bank
getblk: add a,a
        add a,a
        ld e,a
        ld d,0
        ld hl,MAN+M_BLK
        add hl,de
        ld a,(hl)
        inc hl
        inc hl
        ld e,(hl)
        inc hl
        ld d,(hl)
        ex de,hl
        ret

load256:
        push de
        call getblk
        pop de
        ld bc,256
        ; fall through
; romcopy: A = bank, HL = offset, DE = dst, BC = length (crosses banks)
romcopy:
        ld (rc_bank),a
        ld (rc_off),hl
        ld (rc_dst),de
        ld (rc_len),bc
.lp:    ld hl,(rc_len)
        ld a,h
        or l
        ret z
        ld a,(rc_bank)
        ld (BANKREG),a
        ld de,(rc_off)
        ld hl,2000h
        or a
        sbc hl,de               ; bytes left in bank
        ld bc,(rc_len)
        push hl
        or a
        sbc hl,bc
        pop hl
        jr c,.n                 ; avail < len -> n = avail
        ld h,b
        ld l,c                  ; n = len
.n:     ld b,h
        ld c,l
        ld hl,(rc_len)
        or a
        sbc hl,bc
        ld (rc_len),hl
        ld hl,WIN
        add hl,de
        ld de,(rc_dst)
        ldir
        ld (rc_dst),de
        ld hl,0
        ld (rc_off),hl
        ld hl,rc_bank
        inc (hl)
        jr .lp

; ram_page2: map page 2 (8000h-BFFFh) to the RAM slot of page 3
;   (primary slot via port A8h; subslot via FFFFh + BIOS SLTTBL if expanded)
ram_page2:
        in a,(0A8h)
        ld d,a
        rlca
        rlca
        and 3
        ld c,a
        ld b,0
        ld hl,EXPTBL
        add hl,bc
        bit 7,(hl)
        jr z,.prim
        ld a,(0FFFFh)
        cpl
        ld b,a
        and 0C0h
        rrca
        rrca
        ld c,a
        ld a,b
        and 0CFh
        or c
        ld (0FFFFh),a
        inc hl                  ; EXPTBL+slot -> SLTTBL+slot: keep the BIOS
        inc hl                  ; copy in sync, or the next inter-slot call
        inc hl                  ; (e.g. a disk ROM's H.TIMI hook) would
        inc hl                  ; switch page 2 away from RAM again
        ld (hl),a
.prim:  ld a,d
        and 0C0h
        rrca
        rrca
        ld c,a
        ld a,d
        and 0CFh
        or c
        out (0A8h),a
        ret

; -----------------------------------------------------------------------------
; splash: SCREEN 2 boot screen with the colour block banner (pre-rendered by
;   docs/make_banner.py) and three text lines in the BIOS font.  Leaves after
;   a key press (which is swallowed) or 180 frames, then restores the colours.
; -----------------------------------------------------------------------------
splash:
        ld hl,(FORCLR)          ; FORCLR, BAKCLR
        ld (sp_clr),hl
        ld a,(BDRCLR)
        ld (sp_clr+2),a
        ld a,15
        ld (FORCLR),a
        ld a,1
        ld (BAKCLR),a
        ld (BDRCLR),a
        call INIGRP
        di
        ld hl,splash_bin        ; banner: patterns, then colours
        ld de,0700h
        ld bc,1280
        call LDIRVM
        ld hl,splash_bin+1280
        ld de,2700h
        ld bc,1280
        call LDIRVM
        ld hl,sp_text
.line:  ld a,(hl)               ; row (0FFh ends)
        cp 0FFh
        jr z,.wait
        inc hl
        ld e,(hl)               ; column
        inc hl
        ld c,(hl)               ; colour byte (fg<<4 | bg)
        inc hl
        ; DE = VRAM pattern address = (row*32 + col) * 8
        ld d,a
        ld a,e
        add a,a
        add a,a
        add a,a
        ld e,a                  ; col*8 (col < 32)
.ch:    ld a,(hl)
        inc hl
        or a
        jr z,.line
        push hl
        push de
        push bc
        ld l,a
        ld h,0
        add hl,hl
        add hl,hl
        add hl,hl
        ld bc,(CGTABL)
        add hl,bc               ; BIOS font glyph
        ld bc,8
        push de
        call LDIRVM
        pop hl
        ld de,2000h
        add hl,de               ; colour table address
        pop bc
        push bc
        ld a,c
        ld bc,8
        call FILVRM
        pop bc
        pop de
        pop hl
        ld a,e
        add a,8
        ld e,a
        jr .ch
.wait:  ei
        ld hl,(JIFFY)
        ld (sp_t0),hl
.w:     call CHSNS
        jr nz,.key
        ld hl,(JIFFY)
        ld de,(sp_t0)
        or a
        sbc hl,de
        ld de,180
        or a
        sbc hl,de
        jr c,.w
        jr .done
.key:   call CHGET
.done:  di
        ld hl,(sp_clr)
        ld (FORCLR),hl
        ld a,(sp_clr+2)
        ld (BDRCLR),a
        ret
sp_text:                        ; row, column, colour, text, 0
        db 14,3,0F1h,"TERNARY RNN ON A PURE Z80",0
        db 16,1,0E1h,"0.79M 2-BIT WEIGHTS  512KB ROM",0
        db 20,9,0A1h,"PRESS ANY KEY",0
        db 0FFh
splash_bin:
        INCBIN "../build/splash.bin"

no_ram: call INITXT
        ld hl,msg_noram
        call puts
.halt:  jr .halt
msg_noram:
        db "MSX-LLM needs 32 KB of RAM.",0
msg_banner:
        db "MSX-LLM v1  By Patrick Kearns",13,10
        db "Ternary RNN, pure Z80. 0.79M 2-bit",13,10
        db "weights in ROM.",13,10
        db 13,10
        db 13,10
        db "Type the start of a story, or just",13,10
        db "press RETURN.",13,10,0
msg_prompt:
        db 13,10,"> ",0

code_end:
        ds 6000h-$,0FFh         ; bank 0 = 8 KB of code
        INCBIN "../build/model_data.bin"
        OUTEND

; =============================================================================
; variables (no bytes emitted)
        ORG 0B200h
MAN:        ds 96
TOPV:       ds 64               ; topv[16] (biased) then topi[16]
WBUF:       ds 16
rng:        ds 2
seed0:      ds 2
greedy:     ds 1
ntok:       ds 2
ngen:       ds 1
fidx:       ds 2
last_tok:   ds 2
plen:       ds 1
tmp_off:    ds 2
save_sp:    ds 2
mv_rows:    ds 2
mv_nch:     ds 1
mv_in:      ds 2
mv_wbank:   ds 1
mv_woff:    ds 2
mv_c:       ds 1
mv_left:    ds 2
mv_y:       ds 2
mv_n:       ds 2
mv_sp:      ds 2
bt_src:     ds 2
bt_page:    ds 1
bt_g:       ds 1
gu_z:       ds 2
gu_n:       ds 2
gu_h:       ds 2
gu_cnt:     ds 2
gu_zv:      ds 1
sm_tlast:   ds 2
sm_cnt:     ds 2
sm_i:       ds 2
sm_sum:     ds 2
vocab_ptr:  ds 2
en_pos:     ds 1
en_rem:     ds 1
en_blen:    ds 1
en_id:      ds 2
en_bid:     ds 2
en_cnt:     ds 2
pt_id:      ds 2
rc_bank:    ds 1
rc_off:     ds 2
rc_dst:     ds 2
rc_len:     ds 2
sp_clr:     ds 3
sp_t0:      ds 2
vars_end:
