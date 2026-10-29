import unittest,tempfile,hashlib,json,struct,sys,subprocess,shutil
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_inputs import preprocess
from verify_models import verify_csr,verify_manifest

class InputAndAssetContracts(unittest.TestCase):
    def test_bgr_padding_and_byte_count(self):
        x=preprocess(Image.new('RGB',(10,10),(255,0,0)))
        self.assertEqual(x.shape,(48,640,3));self.assertEqual(x.nbytes,368640);np.testing.assert_array_equal(x[0,0],[-1,-1,1]);self.assertTrue((x[:,48:]==0).all())
    def test_wide_crop_caps_width(self):
        x=preprocess(Image.new('RGB',(2000,10),(255,255,255)));self.assertTrue((x==1).all())
    def test_checksum_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'a').write_bytes(b'known');(p/'SHA256SUMS.json').write_text(json.dumps({'a':hashlib.sha256(b'known').hexdigest()}));self.assertEqual(verify_manifest(p),1);(p/'a').write_bytes(b'tampered')
            with self.assertRaises(ValueError):verify_manifest(p)
    def test_manifest_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'SHA256SUMS.json').write_text(json.dumps({'../outside':'0'*64}))
            with self.assertRaises(ValueError):verify_manifest(p)
    def test_csr_bounds_before_gpu_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'sparse';blob=struct.pack('<3I5IIf8f',4,8,1,0,1,1,1,1,2,.1,*([0]*8));p.write_bytes(blob);verify_csr(p,4,8,np.zeros(8,np.float32));bad=bytearray(blob);struct.pack_into('<I',bad,12+5*4,8);p.write_bytes(bad)
            with self.assertRaises(ValueError):verify_csr(p,4,8,np.zeros(8,np.float32))
    @unittest.skipUnless(shutil.which('bash') and sys.platform!='win32','CLI rejection checks require native Linux bash')
    def test_invalid_cli_fails_before_device_access(self):
        run=Path(__file__).resolve().parents[1]/'run.sh'
        for arguments in [['invalid','x','y'],['quality','x','y','311'],['quality','x','y','0'],[]]:
            r=subprocess.run(['bash',str(run),*arguments],capture_output=True);self.assertEqual(r.returncode,2,r.stderr)
if __name__=='__main__':unittest.main()
