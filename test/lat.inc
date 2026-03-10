 enum LatSpecial {
 Spec1 = 1, // (not *64)
 // F2I: 13
 Spec2, // (not F64)
 // F2F: 13
 // FRND: 14
 // I2F: 13
 Spec3, // (with Rb)
 // NANOSLEEP: 19
 // NANOTRAP: 18
 // WARPSYNC: 18
 Spec4, // .168128@.168256@.88128
 // BMMA: 11 11 10
 Spec5, // .16816.F16.*@.1688.F16.*@.F32.{16816.F16|16816.E8M7|1688.E8M10}@.F32.{1684.E8M10|1688.F16|1688.E8M7}@.SP.16816.F16.*@.SP.16832.F16.*@.SP.F32.{16816.E8M10|16832.F16|16832.E8M7}@.SP.F32.{16816.F16|16816.E8M7|1688.E8M10}
 // HMMA: 16 15 42 40 19 19 46 45
 Spec6, // .16816@.16832@.SF@.SF.SP@.SP.16832@.SP.16864
 // QMMA: 10 10 7 10 13 13
 Spec7, // .16816@.1684@.1688
 // DMMA: 10 9 9
 Spec8, // .1CTA@.1CTA.FLUSH@.2CTA
 // UTCBAR: 11 17 11
 Spec9, // .1CTA@.2CTA
 // UTCCP: 10 10
 // UTCHMMA: 12 12
 // UTCIMMA: 12 12
 // UTCMXQMMA: 14 14
 // UTCOMMA: 12 12
 // UTCQMMA: 12 12
 // UTCSHIFT: 13 13
 Spec10, // .32@.32.READ@.64
 // RPCMOV: 9 14 9
 Spec11, // .32@.64
 // CS2R: 7 7
 Spec12, // .64
 // UIMNMX: 9
 // UISETP: 9
 // UMEMSETS: 11
 // USEL: 7
 Spec13, // .64@.64.HI@.64.LO
 // IADD2: 8 11 11
 // IMNMX: 8 11 11
 // ISETP: 8 11 11
 // MOV: 6 9 9
 // SEL: 6 9 9
 Spec14, // .64@.64.HI@.64.LO@.XOR
 // IADD: 7 10 10 8
 Spec15, // .ARRIVE.A1T0.A0T1@.ARRIVE.A1TR.ART0.A0TR.A0TX@.CAS.EXCH@.CCTL@.FLUSH@.LD.NON_UNIFORM@.LD.UNIFORM@.PHASECHK
 // SYNCS: 22 32 14 10 11 20 16 14
 Spec16, // .ARRIVE.A1T0@.ARRIVE.MULTICAST.A1T0
 // SYNCSU: 18 28
 Spec17, // .ARRIVE@.ARRIVE.MULTICAST@.CAS.EXCH@.CCTL@.LD@.PHASECHK
 // USYNCS: 13 23 15 11 9 15
 Spec18, // .ARRIVE@.DEPBAR@.WAIT
 // WARPGROUP: 16 16 14
 Spec19, // .C@.C.LDC@.C.LDC.IVALL@.E.LDC@.I@.LDCU@.QFAULT
 // CCTL: 6 10 16 10 6 9 11
 Spec20, // .DRAIN@.PAUSE
 // BPT: 9 9
 Spec21, // .EX2@.EX2.{F16x2, BF16x2}@.RCP@.RSQ@.TANH
 // MUFU: 8 24 8 8 9
 Spec22, // .F32
 // HADD2: 9
 Spec23, // .FINAL
 // OUT: 9
 Spec24, // .FLUSH
 // USETSHMSZ: 15
 Spec25, // .G@.S@.T
 // FENCE: 7 7 7
 Spec26, // .GSB
 // BGMMA: 9
 // HGMMA: 9
 // IGMMA: 9
 // QGMMA: 9
 Spec27, // .HI@.WIDE@.WIDE.READ.AB@.WIDE.READ.CH@.WIDE.READ.CL@.WIDE.WRITE.DH@.WIDE.WRITE.DL
 // IMAD: 7 9 17 17 17 18 18
 Spec28, // .IMM
 // BRA: 7
 // JMP: 7
 Spec29, // .RELEASE
 // USETMAXREG: 18
 Spec30, // .SFB
 // TEX: 7
 // TLD: 7
 // TLD4: 8
 // TXD: 7
 Spec31, // .SP
 // OMMA: 7
 Spec32, // .SP.{168128.*|16864.*8.*8}@.SP.{16832.*|16864.*4.*4}@.SP.{8832.*|8864.*}@.{16816.*|16832.*4.*4}@.{16832.*8.*8|16864.*}@.{8816.*|8832.*}
 // IMMA: 30 29 23 26 26 20
 Spec33, // .SYNC.DEFER_BLOCKING
 // BAR: 23
 Spec34, // .WIDE
 // IMUL: 9
 // IMUL32I: 12
};
 static const std::unordered_map<std::string_view, std::pair<unsigned char, unsigned char> > s_lats = {
 { "ACQBULK"sv, { 7, 0 } },
 { "ACQSHMINIT"sv, { 10, 0 } },
 { "AL2P"sv, { 4, 0 } },
 { "AL2P_INDEXED"sv, { 12, 0 } },
 { "ALD"sv, { 3, 0 } },
 { "ARRIVES"sv, { 7, 0 } },
 { "AST"sv, { 3, 0 } },
 { "ATOM"sv, { 4, 0 } },
 { "ATOMG"sv, { 5, 0 } },
 { "ATOMS"sv, { 5, 0 } },
 { "B2R"sv, { 3, 0 } },
 { "BAR"sv, { 3, 33 } }, // .SYNC.DEFER_BLOCKING
 { "BGMMA"sv, { 5, 26 } }, // .GSB
 { "BITEXTRACT"sv, { 10, 0 } },
 { "BMMA"sv, { 4, 4 } }, // .168128@.168256@.88128
 { "BMOV"sv, { 4, 0 } },
 { "BMSK"sv, { 4, 0 } },
 { "BPT"sv, { 3, 20 } }, // .DRAIN@.PAUSE
 { "BRA"sv, { 3, 28 } }, // .IMM
 { "BREAK"sv, { 5, 0 } },
 { "BREV"sv, { 4, 0 } },
 { "BRX"sv, { 3, 0 } },
 { "BRXU"sv, { 4, 0 } },
 { "BSSY"sv, { 4, 0 } },
 { "BSYNC"sv, { 5, 0 } },
 { "CALL"sv, { 4, 0 } },
 { "CCTL"sv, { 4, 19 } }, // .C@.C.LDC@.C.LDC.IVALL@.E.LDC@.I@.LDCU@.QFAULT
 { "CCTLL"sv, { 5, 0 } },
 { "CCTLT"sv, { 5, 0 } },
 { "CGAERRBAR"sv, { 9, 0 } },
 { "CLMAD"sv, { 5, 0 } },
 { "CREDUX"sv, { 6, 0 } },
 { "CS2R"sv, { 4, 11 } }, // .32@.64
 { "CS2UR"sv, { 5, 0 } },
 { "CSMTEST"sv, { 7, 0 } },
 { "DADD"sv, { 4, 0 } },
 { "DECOMPRESS"sv, { 10, 0 } },
 { "DEPBAR"sv, { 6, 0 } },
 { "DFMA"sv, { 4, 0 } },
 { "DMMA"sv, { 4, 7 } }, // .16816@.1684@.1688
 { "DMUL"sv, { 4, 0 } },
 { "DPCPMTRIG"sv, { 9, 0 } },
 { "DSETP"sv, { 5, 0 } },
 { "ELECT"sv, { 5, 0 } },
 { "ENDCOLLECTIVE"sv, { 13, 0 } },
 { "ERRBAR"sv, { 6, 0 } },
 { "EXIT"sv, { 4, 0 } },
 { "F2F"sv, { 3, 2 } }, // (not F64)
 { "F2FP"sv, { 4, 0 } },
 { "F2I"sv, { 3, 1 } }, // (not *64)
 { "F2IP"sv, { 4, 0 } },
 { "FADD"sv, { 4, 0 } },
 { "FADD2"sv, { 5, 0 } },
 { "FADD32I"sv, { 7, 0 } },
 { "FCHK"sv, { 4, 0 } },
 { "FENCE"sv, { 0, 25 } }, // .G@.S@.T
 { "FFMA"sv, { 4, 0 } },
 { "FFMA2"sv, { 5, 0 } },
 { "FFMA32I"sv, { 7, 0 } },
 { "FHADD"sv, { 5, 0 } },
 { "FHADD2"sv, { 6, 0 } },
 { "FHFMA"sv, { 5, 0 } },
 { "FHFMA2"sv, { 6, 0 } },
 { "FHMUL2"sv, { 6, 0 } },
 { "FLO"sv, { 3, 0 } },
 { "FMNMX"sv, { 5, 0 } },
 { "FMNMX3"sv, { 6, 0 } },
 { "FMUL"sv, { 4, 0 } },
 { "FMUL2"sv, { 5, 0 } },
 { "FMUL32I"sv, { 7, 0 } },
 { "FOOTPRINT"sv, { 9, 0 } },
 { "FRND"sv, { 4, 2 } }, // (not F64)
 { "FSEL"sv, { 4, 0 } },
 { "FSET"sv, { 4, 0 } },
 { "FSETP"sv, { 5, 0 } },
 { "FSWZADD"sv, { 7, 0 } },
 { "GATHER"sv, { 6, 0 } },
 { "GENMETADATA"sv, { 11, 0 } },
 { "GETLMEMBASE"sv, { 11, 0 } },
 { "HADD2"sv, { 5, 22 } }, // .F32
 { "HADD2_32I"sv, { 9, 0 } },
 { "HFMA2"sv, { 5, 0 } },
 { "HFMA2.MMA"sv, { 9, 0 } },
 { "HFMA2_32I"sv, { 9, 0 } },
 { "HGMMA"sv, { 5, 26 } }, // .GSB
 { "HMMA"sv, { 4, 5 } }, // .16816.F16.*@.1688.F16.*@.F32.{16816.F16|16816.E8M7|1688.E8M10}@.F32.{1684.E8M10|1688.F16|1688.E8M7}@.SP.16816.F16.*@.SP.16832.F16.*@.SP.F32.{16816.E8M10|16832.F16|16832.E8M7}@.SP.F32.{16816.F16|16816.E8M7|1688.E8M10}
 { "HMNMX2"sv, { 6, 0 } },
 { "HMUL2"sv, { 5, 0 } },
 { "HMUL2_32I"sv, { 9, 0 } },
 { "HSET2"sv, { 5, 0 } },
 { "HSETP2"sv, { 6, 0 } },
 { "I2F"sv, { 3, 2 } }, // (not F64)
 { "I2FP"sv, { 4, 0 } },
 { "I2I"sv, { 3, 0 } },
 { "I2IP"sv, { 4, 0 } },
 { "IABS"sv, { 4, 0 } },
 { "IADD"sv, { 4, 14 } }, // .64@.64.HI@.64.LO@.XOR
 { "IADD2"sv, { 5, 13 } }, // .64@.64.HI@.64.LO
 { "IADD3"sv, { 5, 0 } },
 { "IADD32I"sv, { 7, 0 } },
 { "IDE"sv, { 3, 0 } },
 { "IDP"sv, { 3, 0 } },
 { "IDP4A"sv, { 5, 0 } },
 { "IGMMA"sv, { 5, 26 } }, // .GSB
 { "IMAD"sv, { 4, 27 } }, // .HI@.WIDE@.WIDE.READ.AB@.WIDE.READ.CH@.WIDE.READ.CL@.WIDE.WRITE.DH@.WIDE.WRITE.DL
 { "IMMA"sv, { 4, 32 } }, // .SP.{168128.*|16864.*8.*8}@.SP.{16832.*|16864.*4.*4}@.SP.{8832.*|8864.*}@.{16816.*|16832.*4.*4}@.{16832.*8.*8|16864.*}@.{8816.*|8832.*}
 { "IMNMX"sv, { 5, 13 } }, // .64@.64.HI@.64.LO
 { "IMUL"sv, { 4, 34 } }, // .WIDE
 { "IMUL32I"sv, { 7, 34 } }, // .WIDE
 { "IPA"sv, { 3, 0 } },
 { "ISBERD"sv, { 6, 0 } },
 { "ISBEWR"sv, { 6, 0 } },
 { "ISCADD"sv, { 6, 0 } },
 { "ISCADD32I"sv, { 9, 0 } },
 { "ISETP"sv, { 5, 13 } }, // .64@.64.HI@.64.LO
 { "JMP"sv, { 3, 28 } }, // .IMM
 { "JMX"sv, { 3, 0 } },
 { "JMXU"sv, { 4, 0 } },
 { "KILL"sv, { 4, 0 } },
 { "LD"sv, { 2, 0 } },
 { "LDC"sv, { 3, 0 } },
 { "LDCU"sv, { 4, 0 } },
 { "LDG"sv, { 3, 0 } },
 { "LDGDEPBAR"sv, { 9, 0 } },
 { "LDGMC"sv, { 5, 0 } },
 { "LDGSTS"sv, { 6, 0 } },
 { "LDL"sv, { 3, 0 } },
 { "LDS"sv, { 3, 0 } },
 { "LDSM"sv, { 4, 0 } },
 { "LDT"sv, { 3, 0 } },
 { "LDTM"sv, { 4, 0 } },
 { "LDTRAM"sv, { 6, 0 } },
 { "LEA"sv, { 3, 0 } },
 { "LEPC"sv, { 4, 0 } },
 { "LOP"sv, { 3, 0 } },
 { "LOP3"sv, { 4, 0 } },
 { "LOP32I"sv, { 6, 0 } },
 { "MATCH"sv, { 5, 0 } },
 { "MEMBAR"sv, { 6, 0 } },
 { "MOV"sv, { 3, 13 } }, // .64@.64.HI@.64.LO
 { "MOV32I"sv, { 6, 0 } },
 { "MOV64IUR"sv, { 8, 0 } },
 { "MOVM"sv, { 4, 0 } },
 { "MUFU"sv, { 4, 21 } }, // .EX2@.EX2.{F16x2, BF16x2}@.RCP@.RSQ@.TANH
 { "MXQMMA"sv, { 6, 0 } },
 { "NANOSLEEP"sv, { 9, 3 } }, // (with Rb)
 { "NANOTRAP"sv, { 8, 3 } }, // (with Rb)
 { "NONE"sv, { 4, 0 } },
 { "NOP"sv, { 3, 0 } },
 { "OMMA"sv, { 4, 31 } }, // .SP
 { "OUT"sv, { 3, 23 } }, // .FINAL
 { "P2R"sv, { 3, 0 } },
 { "PIXLD"sv, { 5, 0 } },
 { "PLOP3"sv, { 5, 0 } },
 { "PMTRIG"sv, { 6, 0 } },
 { "POPC"sv, { 4, 0 } },
 { "PREEXIT"sv, { 7, 0 } },
 { "PRMT"sv, { 4, 0 } },
 { "PSETP"sv, { 5, 0 } },
 { "QADD4"sv, { 5, 0 } },
 { "QFMA4"sv, { 5, 0 } },
 { "QGMMA"sv, { 5, 26 } }, // .GSB
 { "QMMA"sv, { 4, 6 } }, // .16816@.16832@.SF@.SF.SP@.SP.16832@.SP.16864
 { "QMUL4"sv, { 5, 0 } },
 { "QSPC"sv, { 4, 0 } },
 { "R2B"sv, { 3, 0 } },
 { "R2P"sv, { 3, 0 } },
 { "R2UR"sv, { 4, 0 } },
 { "RED"sv, { 3, 0 } },
 { "REDAS"sv, { 5, 0 } },
 { "REDG"sv, { 4, 0 } },
 { "REDUX"sv, { 5, 0 } },
 { "RET"sv, { 3, 0 } },
 { "RPCMOV"sv, { 6, 10 } }, // .32@.32.READ@.64
 { "RTT"sv, { 3, 0 } },
 { "S2R"sv, { 3, 0 } },
 { "S2UR"sv, { 4, 0 } },
 { "SCATTER"sv, { 7, 0 } },
 { "SEL"sv, { 3, 13 } }, // .64@.64.HI@.64.LO
 { "SETCTAID"sv, { 8, 0 } },
 { "SETLMEMBASE"sv, { 11, 0 } },
 { "SGXT"sv, { 4, 0 } },
 { "SHF"sv, { 3, 0 } },
 { "SHFL"sv, { 4, 0 } },
 { "SHL"sv, { 3, 0 } },
 { "SHR"sv, { 3, 0 } },
 { "SPARSIFY"sv, { 8, 0 } },
 { "SPMETADATA"sv, { 10, 0 } },
 { "ST"sv, { 2, 0 } },
 { "STAS"sv, { 4, 0 } },
 { "STG"sv, { 3, 0 } },
 { "STL"sv, { 3, 0 } },
 { "STS"sv, { 3, 0 } },
 { "STSM"sv, { 4, 0 } },
 { "STT"sv, { 3, 0 } },
 { "STTM"sv, { 4, 0 } },
 { "SUATOM"sv, { 6, 0 } },
 { "SULD"sv, { 4, 0 } },
 { "SUQUERY"sv, { 7, 0 } },
 { "SURED"sv, { 5, 0 } },
 { "SUST"sv, { 4, 0 } },
 { "SYNCS"sv, { 0, 15 } }, // .ARRIVE.A1T0.A0T1@.ARRIVE.A1TR.ART0.A0TR.A0TX@.CAS.EXCH@.CCTL@.FLUSH@.LD.NON_UNIFORM@.LD.UNIFORM@.PHASECHK
 { "SYNCSU"sv, { 0, 16 } }, // .ARRIVE.A1T0@.ARRIVE.MULTICAST.A1T0
 { "TATOMG"sv, { 6, 0 } },
 { "TEX"sv, { 3, 30 } }, // .SFB
 { "TLD"sv, { 3, 30 } }, // .SFB
 { "TLD4"sv, { 4, 30 } }, // .SFB
 { "TMML"sv, { 4, 0 } },
 { "TTUCCTL"sv, { 7, 0 } },
 { "TTUCLOSE"sv, { 8, 0 } },
 { "TTUGO"sv, { 5, 0 } },
 { "TTULD"sv, { 5, 0 } },
 { "TTULD_CLOSE"sv, { 11, 0 } },
 { "TTUMACROFUSE"sv, { 12, 0 } },
 { "TTUOPEN"sv, { 7, 0 } },
 { "TTUST"sv, { 5, 0 } },
 { "TXD"sv, { 3, 30 } }, // .SFB
 { "TXQ"sv, { 3, 0 } },
 { "UACCELCMD"sv, { 9, 0 } },
 { "UBLKCP"sv, { 6, 0 } },
 { "UBLKL2CCTL"sv, { 10, 0 } },
 { "UBLKPF"sv, { 6, 0 } },
 { "UBLKRED"sv, { 7, 0 } },
 { "UBMSK"sv, { 5, 0 } },
 { "UBREV"sv, { 5, 0 } },
 { "UCCTL"sv, { 5, 0 } },
 { "UCGABAR_ARV"sv, { 11, 0 } },
 { "UCGABAR_GET"sv, { 11, 0 } },
 { "UCGABAR_SET"sv, { 11, 0 } },
 { "UCGABAR_WAIT"sv, { 12, 0 } },
 { "UCLEA"sv, { 5, 0 } },
 { "UDLCBAR"sv, { 7, 0 } },
 { "UDLCCP"sv, { 6, 0 } },
 { "UDLCFLUSH"sv, { 9, 0 } },
 { "UDLCHMMA"sv, { 8, 0 } },
 { "UDLCIMMA"sv, { 8, 0 } },
 { "UDLCQMMA"sv, { 8, 0 } },
 { "UDPCBLKCP"sv, { 9, 0 } },
 { "UDPCBLKL2CCTL"sv, { 13, 0 } },
 { "UDPCBLKRED"sv, { 10, 0 } },
 { "UDPCRM"sv, { 6, 0 } },
 { "UDPCTMACCTL"sv, { 11, 0 } },
 { "UDPCTMAL2CCTL"sv, { 13, 0 } },
 { "UDPCTMALDG"sv, { 10, 0 } },
 { "UDPCTMAREDG"sv, { 11, 0 } },
 { "UDPCTMASTG"sv, { 10, 0 } },
 { "UF2F"sv, { 4, 0 } },
 { "UF2FP"sv, { 5, 0 } },
 { "UF2I"sv, { 4, 0 } },
 { "UF2IP"sv, { 5, 0 } },
 { "UFADD"sv, { 5, 0 } },
 { "UFFMA"sv, { 5, 0 } },
 { "UFHADD"sv, { 6, 0 } },
 { "UFHFMA"sv, { 6, 0 } },
 { "UFLO"sv, { 4, 0 } },
 { "UFMNMX"sv, { 6, 0 } },
 { "UFMUL"sv, { 5, 0 } },
 { "UFRND"sv, { 5, 0 } },
 { "UFSEL"sv, { 5, 0 } },
 { "UFSET"sv, { 5, 0 } },
 { "UFSETP"sv, { 6, 0 } },
 { "UGETNEXTWORKID"sv, { 14, 0 } },
 { "UI2F"sv, { 4, 0 } },
 { "UI2FP"sv, { 5, 0 } },
 { "UI2I"sv, { 4, 0 } },
 { "UI2IP"sv, { 5, 0 } },
 { "UIABS"sv, { 5, 0 } },
 { "UIADD3"sv, { 6, 0 } },
 { "UIADD3.64"sv, { 9, 0 } },
 { "UIMAD"sv, { 5, 0 } },
 { "UIMNMX"sv, { 6, 12 } }, // .64
 { "UISETP"sv, { 6, 12 } }, // .64
 { "ULDC"sv, { 4, 0 } },
 { "ULEA"sv, { 4, 0 } },
 { "ULEPC"sv, { 5, 0 } },
 { "ULOP"sv, { 4, 0 } },
 { "ULOP3"sv, { 5, 0 } },
 { "ULOP32I"sv, { 7, 0 } },
 { "UMEMSETS"sv, { 0, 12 } }, // .64
 { "UMOV"sv, { 4, 0 } },
 { "UMOV.64"sv, { 7, 0 } },
 { "UP2UR"sv, { 5, 0 } },
 { "UPLOP3"sv, { 6, 0 } },
 { "UPOPC"sv, { 5, 0 } },
 { "UPRMT"sv, { 5, 0 } },
 { "UPSETP"sv, { 6, 0 } },
 { "UR2UP"sv, { 5, 0 } },
 { "UREDGR"sv, { 6, 0 } },
 { "UREGPRERELEASE"sv, { 14, 0 } },
 { "USEL"sv, { 4, 12 } }, // .64
 { "USETMAXREG"sv, { 10, 29 } }, // .RELEASE
 { "USETSHMSZ"sv, { 9, 24 } }, // .FLUSH
 { "USGXT"sv, { 5, 0 } },
 { "USHF"sv, { 4, 0 } },
 { "USHL"sv, { 4, 0 } },
 { "USHR"sv, { 4, 0 } },
 { "USTGR"sv, { 5, 0 } },
 { "USYNCS"sv, { 0, 17 } }, // .ARRIVE@.ARRIVE.MULTICAST@.CAS.EXCH@.CCTL@.LD@.PHASECHK
 { "UTCATOMSWS"sv, { 10, 0 } },
 { "UTCBAR"sv, { 0, 8 } }, // .1CTA@.1CTA.FLUSH@.2CTA
 { "UTCCP"sv, { 0, 9 } }, // .1CTA@.2CTA
 { "UTCHMMA"sv, { 0, 9 } }, // .1CTA@.2CTA
 { "UTCIMMA"sv, { 0, 9 } }, // .1CTA@.2CTA
 { "UTCLDSWS"sv, { 8, 0 } },
 { "UTCMXQMMA"sv, { 0, 9 } }, // .1CTA@.2CTA
 { "UTCOMMA"sv, { 0, 9 } }, // .1CTA@.2CTA
 { "UTCQMMA"sv, { 0, 9 } }, // .1CTA@.2CTA
 { "UTCSHIFT"sv, { 0, 9 } }, // .1CTA@.2CTA
 { "UTCSTSWS"sv, { 8, 0 } },
 { "UTMACCTL"sv, { 8, 0 } },
 { "UTMACMDFLUSH"sv, { 12, 0 } },
 { "UTMAL2CCTL"sv, { 10, 0 } },
 { "UTMALDG"sv, { 7, 0 } },
 { "UTMALDSP"sv, { 8, 0 } },
 { "UTMAPF"sv, { 6, 0 } },
 { "UTMAREDG"sv, { 8, 0 } },
 { "UTMASTG"sv, { 7, 0 } },
 { "UTMASTSP"sv, { 8, 0 } },
 { "UTRACEEVENT"sv, { 11, 0 } },
 { "UVIADD"sv, { 6, 0 } },
 { "UVIMNMX"sv, { 7, 0 } },
 { "UVIRTCOUNT"sv, { 10, 0 } },
 { "VABSDIFF"sv, { 8, 0 } },
 { "VABSDIFF4"sv, { 9, 0 } },
 { "VHMNMX"sv, { 6, 0 } },
 { "VIADD"sv, { 5, 0 } },
 { "VIADDMNMX"sv, { 9, 0 } },
 { "VIMNMX"sv, { 6, 0 } },
 { "VIMNMX3"sv, { 7, 0 } },
 { "VISET"sv, { 5, 0 } },
 { "VOTE"sv, { 4, 0 } },
 { "VOTEU"sv, { 5, 0 } },
 { "WARPGROUP"sv, { 0, 18 } }, // .ARRIVE@.DEPBAR@.WAIT
 { "WARPGROUPSET"sv, { 12, 0 } },
 { "WARPSYNC"sv, { 8, 3 } }, // (with Rb)
 { "XMAD"sv, { 4, 0 } },
 { "YIELD"sv, { 5, 0 } },
};
