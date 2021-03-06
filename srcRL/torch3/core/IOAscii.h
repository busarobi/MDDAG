// Copyright (C) 2003--2004 Ronan Collobert (collober@idiap.ch)
//                
// This file is part of Torch 3.1.
//
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef IO_ASCII_INC
#define IO_ASCII_INC

#include "IOSequence.h"
#include "DiskXFile.h"

namespace Torch {

/** Handles the standard Ascii sequence format in Torch.
    The format is the following:
    \begin{itemize}
      \item First line should contain the number of frames and the frame size (two int
      separated by a space) of the file.
      \item Next lines are frames of the sequence. One frame is one line. (number_of_columns real
      at each line, separated by a space).
    \end{itemize}

    @author Ronan Collobert (collober@idiap.ch)
*/
class IOAscii : public IOSequence
{
  private:
    DiskXFile *file;
    int current_frame_index;

  public:
    bool one_file_is_one_sequence;
    int n_total_frames;
    char *filename;
    int max_load;
    
    /** Reads the sequence contained in #filename#.
        If #one_file_is_one_sequence# is false, #getSequence()# will return one sequence
        with one frame at each call. (If calling #getSequence(t, foo)#,
        it will put in the sequence #foo# the frame corresponding to the line #t# of the file).
        Note also that if #one_file_is_one_sequence# is false, the access to the IO must be
        sequential when calling #getSequence()#.
        If #max_load_# is positive, it loads only the first #max_load_# frames, if #one_file_is_one_sequence#
        is false.
        The file will be opened when reading the first sequence, and closed when reading the
        last one.
     */
    IOAscii(const char *filename_, bool one_file_is_one_sequence_=false, int max_load_=-1);

    /// Saves one #sequence# in #file# using the ascii format.
    static void saveSequence(XFile *file, Sequence *sequence);

    virtual void getSequence(int t, Sequence *sequence);
    virtual int getNumberOfFrames(int t);
    virtual int getTotalNumberOfFrames();

    virtual ~IOAscii();
};

}

#endif
