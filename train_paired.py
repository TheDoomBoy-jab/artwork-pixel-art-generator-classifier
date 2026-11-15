"""Train a compact conditional GAN on a local manifest of real paired images."""
import argparse
import json
import os
from pathlib import Path
import time
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
from PIL import Image
import tensorflow as tf


def build_models():
    source=tf.keras.Input((64,64,3))
    a=tf.keras.layers.Conv2D(32,4,2,padding='same',activation='leaky_relu')(source)
    b=tf.keras.layers.Conv2D(64,4,2,padding='same',activation='leaky_relu')(a)
    c=tf.keras.layers.Conv2D(128,4,2,padding='same',activation='relu')(b)
    x=tf.keras.layers.Conv2DTranspose(64,4,2,padding='same',activation='relu')(c)
    x=tf.keras.layers.Concatenate()([x,b])
    x=tf.keras.layers.Conv2DTranspose(32,4,2,padding='same',activation='relu')(x)
    x=tf.keras.layers.Concatenate()([x,a])
    output=tf.keras.layers.Conv2DTranspose(3,4,2,padding='same',activation='tanh')(x)
    generator=tf.keras.Model(source,output)
    target=tf.keras.Input((64,64,3));x=tf.keras.layers.Concatenate()([source,target])
    x=tf.keras.layers.Conv2D(32,4,2,padding='same',activation='leaky_relu')(x)
    x=tf.keras.layers.Conv2D(64,4,2,padding='same',activation='leaky_relu')(x)
    discriminator=tf.keras.Model([source,target],tf.keras.layers.Conv2D(1,3,padding='same')(x))
    return generator,discriminator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path('results/paired'))
    parser.add_argument('--epochs',type=int,default=20);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();tf.keras.utils.set_random_seed(args.seed);tf.config.threading.set_intra_op_parallelism_threads(4)
    meta=json.loads((args.data/'manifest.json').read_text());pairs=meta['pairs'];rng=np.random.default_rng(args.seed)
    # Group exact duplicate image pairs instead of splitting duplicate records.
    groups={}
    for row in pairs:groups.setdefault((row['sketch_sha256'],row['art_sha256']),[]).append(row)
    keys=list(groups);rng.shuffle(keys);a=int(len(keys)*.7);b=int(len(keys)*.85)
    manifest={name:[r for key in part for r in groups[key]] for name,part in [('train',keys[:a]),('validation',keys[a:b]),('test',keys[b:])]}
    def load(rows,key):
        return np.stack([np.asarray(Image.open(args.data/r[key]).convert('RGB').resize((64,64)),dtype='float32')/127.5-1 for r in rows])
    arrays={k:(load(v,'sketch'),load(v,'art')) for k,v in manifest.items()}
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'split.json').write_text(json.dumps(manifest,indent=2))
    generator,discriminator=build_models();go=tf.keras.optimizers.Adam(2e-4,beta_1=.5);do=tf.keras.optimizers.Adam(2e-4,beta_1=.5)
    bce=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    @tf.function
    def train(X,Y):
        with tf.GradientTape() as gt,tf.GradientTape() as dt:
            generated=generator(X,training=True);real=discriminator([X,Y],training=True);fake=discriminator([X,generated],training=True)
            gl=bce(tf.ones_like(fake),fake)+100*tf.reduce_mean(tf.abs(Y-generated))
            dl=bce(tf.ones_like(real),real)+bce(tf.zeros_like(fake),fake)
        tf.debugging.assert_all_finite(gl,'Non-finite generator loss');tf.debugging.assert_all_finite(dl,'Non-finite discriminator loss')
        go.apply_gradients(zip(gt.gradient(gl,generator.trainable_variables),generator.trainable_variables))
        do.apply_gradients(zip(dt.gradient(dl,discriminator.trainable_variables),discriminator.trainable_variables));return gl,dl
    def predict(X):return np.concatenate([generator(X[i:i+8],training=False).numpy() for i in range(0,len(X),8)])
    start=time.monotonic();history=[];best=float('inf');X,Y=arrays['train'];vx,vy=arrays['validation']
    for epoch in range(args.epochs):
        losses=[]
        for idx in np.array_split(rng.permutation(len(X)),int(np.ceil(len(X)/8))):losses.append([float(v) for v in train(X[idx],Y[idx])])
        validation=float(np.abs(predict(vx)-vy).mean()/2)
        record={'epoch':epoch+1,'generator_loss':float(np.mean(losses,axis=0)[0]),'discriminator_loss':float(np.mean(losses,axis=0)[1]),'validation_mae_0_1':validation};history.append(record);print(json.dumps(record),flush=True)
        if validation<best:best=validation;generator.save(args.output/'generator.keras')
    generator=tf.keras.models.load_model(args.output/'generator.keras');X,Y=arrays['test'];pred=predict(X);p01=np.clip((pred+1)/2,0,1);y01=(Y+1)/2
    mse=float(np.mean((p01-y01)**2));baseline=np.mean(arrays['train'][1],axis=0)
    metrics={'dataset':meta['dataset'],'selection':meta.get('selection'),'seed':args.seed,'epochs':args.epochs,'counts':{k:len(v) for k,v in manifest.items()},'test_mae_0_1':float(np.abs(p01-y01).mean()),'test_psnr_db':float(-10*np.log10(max(mse,1e-12))),'test_ssim':float(tf.reduce_mean(tf.image.ssim(p01,y01,max_val=1)).numpy()),'mean_training_image_baseline_mae_0_1':float(np.abs(baseline-Y).mean()/2),'history':history,'seconds':time.monotonic()-start,'note':'Exploratory paired-domain translation subset, not original pixel-art collection; no FID or perceptual realism claim'}
    reloaded=tf.keras.models.load_model(args.output/'generator.keras');np.testing.assert_allclose(reloaded(X[:1]).numpy(),generator(X[:1]).numpy(),atol=1e-6);metrics['checkpoint_reload_verified']=True
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(args.output/'examples.npz',source=X[:4],target=Y[:4],prediction=pred[:4]);print(json.dumps({k:v for k,v in metrics.items() if k!='history'}),flush=True)

if __name__=='__main__':main()
