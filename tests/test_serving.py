import unittest
from app import create_app
from serving import load_artifact,recommend

class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=load_artifact();cls.client=create_app(cls.model).test_client()
    def test_cold_start_and_exclusions(self):
        original=recommend(self.model,limit=6)
        self.assertEqual(len(original),6)
        self.assertEqual(len({p['id'] for p in original}),6)
        seen=[p['id'] for p in original[:2]]
        ranked=recommend(self.model,liked_ids=seen,excluded_ids=[original[2]['id']])
        self.assertFalse(set(seen+[original[2]['id']])&{p['id'] for p in ranked})
    def test_api_matches_core(self):
        liked=[self.model['products'][0]['id']]
        response=self.client.post('/api/recommend',json={'liked_ids':liked,'limit':5})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json['recommendations'],recommend(self.model,liked,limit=5))
    def test_rejects_invalid_inputs(self):
        for payload in [[],{'limit':True},{'limit':0},{'liked_ids':['unknown']},{'liked_ids':'oops'},{'query':[]}]:
            self.assertEqual(self.client.post('/api/recommend',json=payload).status_code,400)
        self.assertEqual(self.client.post('/api/recommend',data='x').status_code,415)
    def test_health_static_and_no_user_data(self):
        self.assertEqual(self.client.get('/health').json['products'],len(self.model['products']))
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/assets/app.mjs').status_code,200)
        self.assertNotIn('reviews_username',str(self.model))
        self.assertNotIn('reviews_text',str(self.model))
    def test_filters_and_ties(self):
        category=self.model['products'][0]['category']
        out=recommend(self.model,category=category,limit=20)
        self.assertTrue(out)
        self.assertTrue(all(p['category']==category for p in out))
        self.assertEqual(recommend(self.model,query='does-not-exist-9345'),[])
        self.assertEqual(recommend(self.model),recommend(self.model))

if __name__=='__main__':unittest.main()
