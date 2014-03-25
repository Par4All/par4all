/*
 * Copyright (c) 1999-2001,2005-2007,2010-2012 Apple Inc. All Rights Reserved.
 *
 * @APPLE_LICENSE_HEADER_START@
 *
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 *
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 *
 * @APPLE_LICENSE_HEADER_END@
 */

#include <stdio.h>
#include <stdint.h>

typedef struct {
    int mocked;
} Mocked;

typedef Mocked SSLPubKey;


/*
 *   Dup'd sslTypes.h
 */

typedef struct
{   size_t  length;
    uint8_t *data;
} SSLBuffer;


/*
 *   Mocked MacTypes.h
 */
typedef unsigned char                   UInt8;
typedef signed char                     SInt8;
typedef unsigned short                  UInt16;
typedef signed short                    SInt16;

#if __LP64__
typedef unsigned int                    UInt32;
typedef signed int                      SInt32;
#else
typedef unsigned long                   UInt32;
typedef signed long                     SInt32;
#endif

typedef SInt32                          OSStatus;
typedef unsigned char                   Boolean;


/*
 *   Mocked sslContext.h
 */
#define SSL_CLIENT_SRVR_RAND_SIZE       32

typedef struct
{
    SSLPubKey           *peerPubKey;
    uint8_t             clientRandom[SSL_CLIENT_SRVR_RAND_SIZE];
    uint8_t             serverRandom[SSL_CLIENT_SRVR_RAND_SIZE];

} SSLContext;


/*
 *    Mocked / Dup'd tls_digest.h
 */ 
#define SSL_MD5_DIGEST_LEN      16
#define SSL_SHA1_DIGEST_LEN     20
#define SSL_SHA256_DIGEST_LEN   32
#define SSL_SHA384_DIGEST_LEN   48
#define SSL_MAX_DIGEST_LEN      48 /* >= SSL_MD5_DIGEST_LEN + SSL_SHA1_DIGEST_LEN */

#define MAX_MAC_PADDING         48  /* MD5 MAC padding size = 48 bytes */

//extern const uint8_t SSLMACPad1[], SSLMACPad2[];

typedef int (*HashInit)(SSLBuffer *digestCtx);
typedef int (*HashUpdate)(SSLBuffer *digestCtx, const SSLBuffer *data);
/* HashFinal also does HashClose */
typedef int (*HashFinal)(SSLBuffer *digestCtx, SSLBuffer *digest);
typedef int (*HashClose)(SSLBuffer *digestCtx);
typedef int (*HashClone)(const SSLBuffer *src, SSLBuffer *dest);

typedef struct
{
    uint32_t    digestSize;
    uint32_t    macPadSize;
    uint32_t    contextSize;
    HashInit    init;
    HashUpdate  update;
    HashFinal   final;
    HashClose   close;
    HashClone   clone;
} HashReference;

extern const HashReference SSLHashNull;
extern const HashReference SSLHashMD5;
extern const HashReference SSLHashSHA1;
extern const HashReference SSLHashSHA256;
extern const HashReference SSLHashSHA384;



/*
 *    Mocked / Dup'd sslDigest.h
 */ 

extern OSStatus ReadyHash(const HashReference *ref, SSLBuffer *state);

/*
 *  Mocked / Dup'd sslMemory.h
 */
extern int SSLFreeBuffer(SSLBuffer *buf);

/*
 *  Mocked sslCrypto.h
 */

extern OSStatus sslRawVerify(
    SSLContext          *ctx,
    SSLPubKey           *pubKey,
    const uint8_t       *plainText,
    size_t              plainTextLen,
    const uint8_t       *sig,
    size_t              sigLen);

/*
 * Misc
 */ 

#define sslErrorLog(x, y) (void)x

