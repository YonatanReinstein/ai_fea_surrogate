IRIT_DIR = /home/ryonatan/irit

include $(IRIT_DIR)/makeflag.unx

OBJS = model.o

all: model

model.o: model.c
	$(CC) $(CFLAGS) $(IRIT_URT_FLAGS) $(IRIT_GIF_FLAGS) \
	$(IRIT_PNG_FLAGS) $(IRIT_FREETYPE_FLAGS) $(IRIT_ROXML_FLAGS) \
	$(IRIT_MORE_FLAGS) -I. -I$(IRIT_INC_DIR) $(IRIT_URT_INC) \
	$(IRIT_GIF_INC) $(IRIT_FREETYPE_INC) $(IRIT_ROXML_INC) \
	$(IRIT_MORE_INC) -o $@ -c $<

model: $(OBJS)
	$(CC) $(CFLAGS) -o model $(OBJS) $(IRIT_LIBS) -lm $(MORELIBS)

clean:
	rm -f *.o model