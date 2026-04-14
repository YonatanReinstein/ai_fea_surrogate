IRIT_DIR = /home/ryonatan/projects/ai_fea_surrogate/irit

include $(IRIT_DIR)/makeflag.unx

OBJS = model.o

all: model

model.o: model_lin.c
	$(CC) $(CFLAGS) $(IRIT_URT_FLAGS) $(IRIT_GIF_FLAGS) \
	$(IRIT_PNG_FLAGS) $(IRIT_FREETYPE_FLAGS) $(IRIT_ROXML_FLAGS) \
	$(IRIT_MORE_FLAGS) -I. -I$(IRIT_INC_DIR) $(IRIT_URT_INC) \
	$(IRIT_GIF_INC) $(IRIT_FREETYPE_INC) $(IRIT_ROXML_INC) \
	$(IRIT_MORE_INC) -o $@ -c $<

model: $(OBJS)
	$(CC) $(CFLAGS) -o model $(OBJS) $(IRIT_DIR)/lib/circlink.oD \
	-L$(IRIT_DIR)/lib -lIritExt -lIritGrap -lIritUser -lIritRndr -lIritBool \
	-lIritPrsr -lIritVMdl -lIritMdl -lIritMvar -lIritTrim -lIritTriv \
	-lIritTrng -lIritSymb -lIritCagd -lIritGeom -lIritMisc -lIritXtra \
	/usr/lib64/libfreetype.so -lz -lgomp -lm

test_trivar: test_trivar.c
	$(CC) $(CFLAGS) $(IRIT_URT_FLAGS) $(IRIT_GIF_FLAGS) \
	$(IRIT_PNG_FLAGS) $(IRIT_FREETYPE_FLAGS) $(IRIT_ROTML_FLAGS) \
	$(IRIT_MORE_FLAGS) -I. -I$(IRIT_INC_DIR) $(IRIT_URT_INC) \
	$(IRIT_GIF_INC) $(IRIT_FREETYPE_INC) $(IRIT_ROTML_INC) \
	$(IRIT_MORE_INC) -o test_trivar $< $(IRIT_DIR)/lib/circlink.oD \
	-L$(IRIT_DIR)/lib -lIritExt -lIritGrap -lIritUser -lIritRndr -lIritBool \
	-lIritPrsr -lIritVMdl -lIritMdl -lIritMvar -lIritTrim -lIritTriv \
	-lIritTrng -lIritSymb -lIritCagd -lIritGeom -lIritMisc -lIritXtra \
	/usr/lib64/libfreetype.so -lz -lgomp -lm

clean:
	rm -f *.o model test_trivar